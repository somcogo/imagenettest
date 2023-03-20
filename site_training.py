import sys
import argparse
import os
import datetime
import shutil
import random

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional

from models.model import ResNet18Model, Encoder, TinySwin, SmallSwin, LargeSwin
from utils.logconf import logging
from utils.data_loader import get_trn_loader, get_val_loader, get_multi_site_trn_loader, get_multi_site_val_loader

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

class MultiSiteTrainingApp:
    def __init__(self, sys_argv=None, epochs=None, batch_size=None, logdir=None, lr=None, site_number=5, comment=None, layer=None, sub_layer=None, model_name=None, merge_mode=None, optimizer_type=None, use_scheduler=None, label_smoothing=None, T_max=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser(description="Test training")
        parser.add_argument("--epochs", default=6, type=int, help="number of training epochs")
        parser.add_argument("--batch_size", default=512, type=int, help="number of batch size")
        parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
        parser.add_argument("--in_channels", default=3, type=int, help="number of image channels")
        parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
        parser.add_argument("--site_number", default=5, type=int, help="number of sites taking part in learning")
        parser.add_argument("--layer", default=None, type=str, help="layer in which training should take place")
        parser.add_argument("--model_name", default='resnet', type=str, help="name of model to use")
        parser.add_argument("--merge_mode", default='projection', type=str, help="describes which parameters of the model to merge")
        parser.add_argument("--optimizer_type", default='adamw', type=str, help="type of optimizer to use")
        parser.add_argument("--use_scheduler", default=False, type=bool, help="determines whether to use LR scheduling or not")
        parser.add_argument("--label_smoothing", default=0.0, type=float, help="label smoothing in Cross Entropy Loss")
        parser.add_argument("--T_max", default=1000, type=int, help="T_max in Cosine LR scheduler")
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
        if layer is not None:
            self.args.layer = layer
        if sub_layer is not None:
            self.args.sub_layer = sub_layer
        if model_name is not None:
            self.args.model_name = model_name
        if merge_mode is not None:
            self.args.merge_mode = merge_mode
        if optimizer_type is not None:
            self.args.optimizer_type = optimizer_type
        if use_scheduler is not None:
            self.args.use_scheduler = use_scheduler
        if label_smoothing is not None:
            self.args.label_smoothing = label_smoothing
        if T_max is not None:
            self.args.T_max = T_max
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
                models.append(TinySwin(num_classes=200))
            elif self.args.model_name == 'swins':
                models.append(SmallSwin(num_classes=200))
            elif self.args.model_name == 'swinl':
                models.append(LargeSwin(num_classes=200))
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
        if self.args.use_scheduler:
            schedulers = []
            for optimizer in self.optims:
                schedulers.append(CosineAnnealingLR(optimizer, T_max=self.args.T_max))
        else:
            schedulers = None
        return schedulers

    def initDl(self):
        trn_dls = []
        for i in range(self.args.site_number):
            trn_dls.append(get_trn_loader(self.args.batch_size, site=0, device=self.device))

        val_dl = get_val_loader(self.args.batch_size, device=self.device)

        multi_trn_dl = get_multi_site_trn_loader(self.args.batch_size, site_number=self.args.site_number)
        multi_val_dl = get_multi_site_val_loader(self.args.batch_size, site_number=self.args.site_number)

        return trn_dls, val_dl, multi_trn_dl, multi_val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        trn_dls, val_dl, multi_trn_dl, multi_val_dl = self.initDl()

        saving_criterion = 0
        validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):
            
            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.args.epochs,
                    len(trn_dls[0]),
                    len(val_dl),
                    self.args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

            trnMetrics = torch.zeros(5, len(trn_dls[0]), device=self.device)

            trnMetrics = self.doMultiTraining(epoch_ndx, multi_trn_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, correct_ratio = self.doMultiValidation(epoch_ndx, multi_val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics)
                saving_criterion = max(correct_ratio, saving_criterion)

                self.saveModel('mnist', epoch_ndx, correct_ratio == saving_criterion)
            
            if self.schedulers is not None:
                for scheduler in self.schedulers:
                    scheduler.step()
                    # log.debug(scheduler.get_last_lr())

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dls):
        trnMetrics = torch.zeros(5, 2 + self.args.site_number, len(train_dls[0]), device=self.device)
        for i in range(self.args.site_number):

            self.models[i].train()

            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.warning('E{} Training on site {} ---/{} starting'.format(epoch_ndx, i,len(train_dls[i])))

            for batch_ndx, batch_tuple in enumerate(train_dls[i]):
                self.optims[i].zero_grad()

                loss, _ = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    trnMetrics[i],
                    self.models[i],
                    'trn')

                loss.backward()
                self.optims[i].step()

            if batch_ndx % 100 == 0:
                log.info('E{} Training {}/{}'.format(epoch_ndx, batch_ndx, len(train_dls[0])))

        self.totalTrainingSamples_count += len(train_dls[0].dataset)

        return trnMetrics.to('cpu')

    def doMultiTraining(self, epoch_ndx, mutli_trn_dl):
        for model in self.models:
            model.train()
        trnMetrics = torch.zeros(2 + self.args.site_number, len(mutli_trn_dl), device=self.device)

        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
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
                self.mergeParams(layer_names=['conv0', 'block1', 'block2', 'block3', 'block4', 'lin'], depth=0)
            elif self.args.merge_mode == 'notattention':
                self.mergeParams(layer_names=['conv0', 'conv1', 'skip', 'weight', 'bias', 'norm0', 'norm1'], depth=1)

        self.totalTrainingSamples_count += len(mutli_trn_dl.dataset) * self.args.site_number
        return trnMetrics.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.models[0].eval()
            valMetrics = torch.zeros(2 + self.args.site_number, len(val_dl), device=self.device)

            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(val_dl)))

            for batch_ndx, batch_tuple in enumerate(val_dl):
                _, accuracy = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    valMetrics,
                    self.models[0],
                    'val'
                )
                if batch_ndx % 100 == 0:
                    log.info('E{} Validation {}/{}'.format(epoch_ndx, batch_ndx, len(val_dl)))

        return valMetrics.to('cpu'), accuracy

    def doMultiValidation(self, epoch_ndx, multi_val_dl):
        with torch.no_grad():
            valMetrics = torch.zeros(2 + self.args.site_number, len(multi_val_dl), device=self.device)
            for model in self.models:
                model.eval()

            if epoch_ndx == 1 or epoch_ndx % 10 == 0:
                log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(multi_val_dl)))

            for batch_ndx, batch_tuples in enumerate(multi_val_dl):

                loss, accuracy = self.computeMultiBatchLoss(
                    batch_ndx,
                    batch_tuples,
                    valMetrics,
                    'val'
                )

        return valMetrics.to('cpu'), accuracy

    def computeBatchLoss(self, batch_ndx, batch_tup, metrics, model, mode):
        batch, labels = batch_tup
        batch = batch.to(device=self.device, non_blocking=True)
        labels = labels.to(device=self.device, non_blocking=True)

        if mode == 'trn':
            angle = random.choice([0, 90, 180, 270])
            flip = random.choice([True, False])
            scale = random.uniform(0.9, 1.1)
            batch = functional.rotate(batch, angle)
            if flip:
                batch = functional.hflip(batch)
            batch = scale * batch

        pred = model(batch)
        pred_label = torch.argmax(pred, dim=1)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        loss = loss_fn(pred, labels)

        correct_mask = pred_label == labels
        correct = torch.sum(correct_mask)
        accuracy = correct / batch.shape[0]

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

    def computeMultiBatchLoss(self, batch_ndx, batch_tups, metrics, mode):
        batches, labels = batch_tups
        batches = batches.to(device=self.device, non_blocking=True).permute(1, 0, 2, 3, 4)
        labels = labels.to(device=self.device, non_blocking=True).permute(1, 0).flatten()

        if mode == 'trn':
            angle = random.choice([0, 90, 180, 270])
            flip = random.choice([True, False])
            scale = random.uniform(0.9, 1.1)
            for batch in batches:
                batch = functional.rotate(batch, angle)
            if flip:
                for batch in batches:
                    batch = functional.hflip(batch)
            batches = scale * batches

        preds = torch.Tensor([]).to(device=self.device)
        for i in range(self.args.site_number):
            preds= torch.cat((preds, self.models[i](batches[i])), 0)
        pred_labels = torch.argmax(preds, dim=1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(preds, labels)
        # preds = preds.view(self.args.site_number, -1, 200)
        # labels = labels.view(self.args.site_number, -1)

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

        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            log.info(
                "E{} {}:{} loss".format(
                    epoch_ndx,
                    mode_str,
                    metrics[0].mean()
                )
            )

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
        dicts = []
        for i in range(self.args.site_number):
            dicts.append(self.models[i].state_dict())
        dict_avg = {}

        names = self.models[0].named_parameters()

        for name, _ in names:
            if self.args.model_name == 'resnet':
                layer = name.split('.')[1]
                sub_layer = name.split('.')[2]
                if (layer != self.args.layer or sub_layer != self.args.sub_layer) and (layer != 'conv1') and (layer != 'fc'):
                    dict_avg[name] = torch.zeros(dicts[0][name].shape, device=self.device)
                    for i in range(self.args.site_number):
                        dict_avg[name] += dicts[i][name]
                    dict_avg[name] = dict_avg[name] / self.args.site_number
            if self.args.model_name == 'unet':
                layer = name.split('.')[depth]
                if layer in layer_names:
                    dict_avg[name] = torch.zeros(dicts[0][name].shape, device=self.device)
                    for i in range(self.args.site_number):
                        dict_avg[name] += dicts[i][name]
                    dict_avg[name] = dict_avg[name] / self.args.site_number
            if self.args.model_name == 'swint':
                dict_avg[name] = torch.zeros(dicts[0][name].shape, device=self.device)
                for i in range(self.args.site_number):
                    dict_avg[name] += dicts[i][name]
                dict_avg[name] = dict_avg[name] / self.args.site_number

        for i in range(self.args.site_number):
            self.models[i].load_state_dict(dict_avg, strict=False)


if __name__ == '__main__':
    MultiSiteTrainingApp().main()
