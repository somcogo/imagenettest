import sys
import argparse
import os
import datetime
import shutil
import random

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional, RandomResizedCrop

from models.model import ResNet18Model, Encoder, TinySwin, SmallSwin, LargeSwin
from utils.logconf import logging
from utils.data_loader import get_multi_site_trn_loader, get_multi_site_val_loader
from utils.ops import aug_image

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class MultiSiteTrainingApp:
    def __init__(self, sys_argv=None, epochs=None, batch_size=None, logdir=None, lr=None, site_number=5, comment=None, model_name=None, merge_mode=None, optimizer_type=None, label_smoothing=None, T_max=None, pretrained=None, aug_mode=None, scheduler_mode=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser(description="Test training")
        parser.add_argument("--epochs", default=6, type=int, help="number of training epochs")
        parser.add_argument("--batch_size", default=512, type=int, help="number of batch size")
        parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
        parser.add_argument("--in_channels", default=3, type=int, help="number of image channels")
        parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
        parser.add_argument("--site_number", default=5, type=int, help="number of sites taking part in learning")
        parser.add_argument("--model_name", default='resnet', type=str, help="name of model to use")
        parser.add_argument("--merge_mode", default='everything', type=str, help="describes which parameters of the model to merge")
        parser.add_argument("--optimizer_type", default='adamw', type=str, help="type of optimizer to use")
        parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing in Cross Entropy Loss")
        parser.add_argument("--T_max", default=1000, type=int, help="T_max in Cosine LR scheduler")
        parser.add_argument("--pretrained", default=False, type=bool, help="use pretrained model")
        parser.add_argument("--aug_mode", default='standard', type=str, help="mode of data augmentation")
        parser.add_argument("--scheduler_mode", default=None, type=str, help="mode of LR scheduling")
        parser.add_argument('comment', help="Comment suffix for Tensorboard run.", nargs='?', default='dwlpt')

        self.args = parser.parse_args()
        if epochs is not None:
            self.args.epochs = epochs
        if batch_size is not None:
            self.args.batch_size = batch_size
        if logdir is not None:
            self.args.logdir = logdir
        if lr is not None:
            self.args.lr = lr
        if site_number is not None:
            self.args.site_number = site_number
        if comment is not None:
            self.args.comment = comment
        if model_name is not None:
            self.args.model_name = model_name
        if merge_mode is not None:
            self.args.merge_mode = merge_mode
        if optimizer_type is not None:
            self.args.optimizer_type = optimizer_type
        if label_smoothing is not None:
            self.args.label_smoothing = label_smoothing
        if T_max is not None:
            self.args.T_max = T_max
        if pretrained is not None:
            self.args.pretrained = pretrained
        if aug_mode is not None:
            self.args.aug_mode = aug_mode
        if scheduler_mode is not None:
            self.args.scheduler_mode = scheduler_mode
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.args.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.models= self.initModel()
        self.optims = self.initOptimizer()
        self.schedulers = self.initScheduler()
        # Parameter check
        # for i in range(site_number):
        #     print(torch.equal(list(self.optims[i].param_groups[0]['params'])[0], list(self.models[i].parameters())[0]))

    def initModel(self):
        models = []
        for i in range(self.args.site_number):
            if self.args.model_name == 'resnet':
                models.append(ResNet18Model(num_classes=200))
            elif self.args.model_name == 'unet':
                models.append(Encoder(num_classes=200))
            elif self.args.model_name == 'swint':
                models.append(TinySwin(num_classes=200, pretrained=self.args.pretrained))
            elif self.args.model_name == 'swins':
                models.append(SmallSwin(num_classes=200, pretrained=self.args.pretrained))
            elif self.args.model_name == 'swinl':
                models.append(LargeSwin(num_classes=200, pretrained=self.args.pretrained))
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                for model in models:
                    model = nn.DataParallel(model)
            for model in models:
                model = model.to(self.device)
        return models

    def initOptimizer(self):
        optims = []
        for i in range(self.args.site_number):
            if self.args.optimizer_type == 'adamw':
                optims.append(AdamW(params=self.models[i].parameters(), lr=self.args.lr, weight_decay=1e-5))
            if self.args.optimizer_type == 'adamwnew':
                optims.append(AdamW(params=self.models[i].parameters(), lr=self.args.lr, weight_decay=0.05))
            if self.args.optimizer_type == 'sgd':
                optims.append(SGD(params=self.models[i].parameters(), lr=self.args.lr, weight_decay=0.0001, momentum=0.9))

        return optims
    
    def initScheduler(self):
        if self.args.scheduler_mode == 'cosine':
            schedulers = []
            for optimizer in self.optims:
                schedulers.append(CosineAnnealingLR(optimizer, T_max=self.args.T_max))
        elif self.args.scheduler_mode == 'onecycle':
            schedulers = []
            images_per_site = 100000 // self.args.site_number
            for optimizer in self.optims:
                schedulers.append(OneCycleLR(optimizer, max_lr=0.01, 
                                             steps_per_epoch=images_per_site,
                                             epochs=self.args.epochs, div_factor=10,
                                             final_div_factor=10, pct_start=10/self.args.epochs))
        else:
            assert self.args.scheduler_mode is None
            schedulers = None
        return schedulers

    def initDl(self):
        multi_trn_dl = get_multi_site_trn_loader(self.args.batch_size, site_number=self.args.site_number)
        multi_val_dl = get_multi_site_val_loader(self.args.batch_size, site_number=self.args.site_number)

        return multi_trn_dl, multi_val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        multi_trn_dl, multi_val_dl = self.initDl()

        saving_criterion = 0
        validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):
            
            if epoch_ndx == 1:
                log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.args.epochs,
                    len(multi_trn_dl),
                    len(multi_val_dl),
                    self.args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trnMetrics = self.doMultiTraining(epoch_ndx, multi_trn_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, correct_ratio = self.doMultiValidation(epoch_ndx, multi_val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics)
                saving_criterion = max(correct_ratio, saving_criterion)

                self.saveModel('mnist', epoch_ndx, correct_ratio == saving_criterion)
                log.info(
                    "E{}/{} trn:{:6.4f} loss val:{:6.4f} loss".format(
                        epoch_ndx,
                        self.args.epochs,
                        trnMetrics[0].mean(),
                        valMetrics[0].mean()
                    )
                )
            
            if self.args.scheduler_mode == 'cosine':
                for scheduler in self.schedulers:
                    scheduler.step()
                    # log.debug(scheduler.get_last_lr())

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doMultiTraining(self, epoch_ndx, mutli_trn_dl):
        for model in self.models:
            model.train()
        trnMetrics = torch.zeros(2 + self.args.site_number, len(mutli_trn_dl), device=self.device)

        if epoch_ndx == 1:
            log.warning('E{} Training ---/{} starting'.format(epoch_ndx, len(mutli_trn_dl)))

        for batch_ndx, batch_tuples in enumerate(mutli_trn_dl):
            for optim in self.optims:
                optim.zero_grad()

            loss, _ = self.computeMultiBatchLoss(
                batch_ndx,
                batch_tuples,
                trnMetrics,
                'trn')
            loss.backward()
            
            for optim in self.optims:
                optim.step()

            assert self.args.merge_mode in ['projection', 'second_half', 'first_half', 'last3/4', 'notnorms', 'attention', 'everything', 'notattention']
            if self.args.merge_mode == 'projection':
                self.mergeParams(layer_names=['qkv'], depth=1)
            elif self.args.merge_mode == 'second_half':
                self.mergeParams(layer_names=['block3', 'block4', 'lin'], depth=0)
            elif self.args.merge_mode == 'first_half':
                self.mergeParams(layer_names=['block1', 'block2', 'conv0'], depth=0)
            elif self.args.merge_mode == 'last3/4':
                self.mergeParams(layer_names=['block2', 'block3', 'block4', 'lin'], depth=0)
            elif self.args.merge_mode == 'notnorms':
                self.mergeParams(layer_names=['conv0', 'conv1', 'skip', 'qkv', 'proj'], depth=1)
            elif self.args.merge_mode == 'attention':
                self.mergeParams(layer_names=['qkv', 'proj'], depth=1)
            elif self.args.merge_mode == 'everything':
                if self.args.model_name == 'unet':
                    self.mergeParams(layer_names=['conv0', 'block1', 'block2', 'block3', 'block4', 'lin'], depth=0)
                elif self.args.model_name == 'resnet':
                    self.mergeParams(layer_names=['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'], depth=1)
            elif self.args.merge_mode == 'notattention':
                self.mergeParams(layer_names=['conv0', 'conv1', 'skip', 'weight', 'bias', 'norm0', 'norm1'], depth=1)

        self.totalTrainingSamples_count += len(mutli_trn_dl.dataset) * self.args.site_number
        return trnMetrics.to('cpu')

    def doMultiValidation(self, epoch_ndx, multi_val_dl):
        with torch.no_grad():
            valMetrics = torch.zeros(2 + self.args.site_number, len(multi_val_dl), device=self.device)
            for model in self.models:
                model.eval()

            if epoch_ndx == 1:
                log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(multi_val_dl)))

            for batch_ndx, batch_tuples in enumerate(multi_val_dl):

                loss, accuracy = self.computeMultiBatchLoss(
                    batch_ndx,
                    batch_tuples,
                    valMetrics,
                    'val'
                )

        return valMetrics.to('cpu'), accuracy

    def computeMultiBatchLoss(self, batch_ndx, batch_tups, metrics, mode):
        batches, labels = batch_tups
        batches = batches.to(device=self.device, non_blocking=True).permute(1, 0, 2, 3, 4)
        labels = labels.to(device=self.device, non_blocking=True).permute(1, 0).flatten()

        if mode == 'trn':
            assert self.args.aug_mode in ['standard', 'random_resized_crop', 'resnet']
            batches = aug_image(batches, self.args.aug_mode, multi_training=True)

        preds = torch.Tensor([]).to(device=self.device)
        for i in range(self.args.site_number):
            preds= torch.cat((preds, self.models[i](batches[i])), 0)
        pred_labels = torch.argmax(preds, dim=1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(preds, labels)

        correct_mask = pred_labels == labels
        correct = torch.sum(correct_mask)
        accuracy = correct / batches.shape[1] / self.args.site_number

        labels_per_site = 200 // self.args.site_number
        accuracy_per_class = []
        for i in range(self.args.site_number):
            class_mask = ((i * labels_per_site) <= labels) & (labels < ((i+1) * labels_per_site))
            correct_per_class = torch.sum(correct_mask[class_mask])
            total_per_class = torch.sum(class_mask)
            accuracy_per_class.append(correct_per_class / total_per_class * 100)

        metrics[0, batch_ndx] = loss.detach()
        metrics[1, batch_ndx] = accuracy.detach() * 100
        metrics[2: self.args.site_number + 2, batch_ndx] = torch.Tensor(accuracy_per_class)
        return loss, accuracy

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics
    ):
        self.initTensorboardWriters()

        writer = getattr(self, mode_str + '_writer')
        writer.add_scalar(
            'loss_total',
            scalar_value=metrics[0].mean(),
            global_step=self.totalTrainingSamples_count
        )
        writer.add_scalar(
            'accuracy/overall',
            scalar_value=metrics[1].mean(),
            global_step=self.totalTrainingSamples_count
        )
        for i in range(self.args.site_number):
            writer.add_scalar(
                'accuracy/class {}'.format(i + 1),
                scalar_value=metrics[2+i].mean(),
                global_step=self.totalTrainingSamples_count
            )
        writer.flush()

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'saved_models',
            self.args.logdir,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.args.comment,
                self.totalTrainingSamples_count
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.models[0]
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optims[0].state_dict(),
            'optimizer_name': type(self.optims[0]).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count
        }

        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'saved_models',
                self.args.logdir,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.args.comment,
                    'best'
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

    def mergeParams(self, layer_names=None, depth=None):
        state_dicts = []
        for model in self.models:
            state_dicts.append(model.state_dict())

        dict_avg = {}
        names = self.models[0].named_parameters()
        for name, _ in names:
            layer = name.split('.')[depth]
            if layer in layer_names:
                dict_avg[name] = torch.zeros(state_dicts[0][name].shape, device=self.device)
                for state_dict in state_dicts:
                    dict_avg[name] += state_dict[name]
                dict_avg[name] = dict_avg[name] / len(state_dicts)

        for model in self.models:
            model.load_state_dict(dict_avg, strict=False)


if __name__ == '__main__':
    MultiSiteTrainingApp().main()
