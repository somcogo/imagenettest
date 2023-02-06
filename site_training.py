import sys
import argparse
import os
import datetime
import shutil
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.transforms import functional

from models.model import ResNet18Model
from utils.logconf import logging
from utils.data_loader import get_trn_loader, get_tst_loader, get_val_loader
from utils.ops import aug_rand
from utils.losses import SampleLoss

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class MultiSiteTrainingApp:
    def __init__(self, sys_argv=None, epochs=None, batch_size=None, logdir=None, lr=None, site_number=None, comment=None, layer=None):
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
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.logdir = os.path.join('./runs', self.args.logdir)
        os.makedirs(self.logdir, exist_ok=True)

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.models= self.initModel()
        self.params_to_update = []
        for i in range(self.args.site_number):
            self.params_to_update.append([])
        if self.args.layer is not None:
            for i in range(self.args.site_number):
                for name, param in self.models[i].named_parameters():
                    if name.split('.')[1] == self.args.layer:
                        self.params_to_update[i].append(param)
        else:
            for i in range(self.args.site_number):
                for name, param in self.models[i].named_parameters():
                    self.params_to_update[i].append(param)


        for i in range(self.args.site_number):
            for param in self.models[i].parameters():
                param.requires_grad = False
        for i in range(self.args.site_number):
            for param in self.params_to_update[i]:
                param.requires_grad = True
        self.optims = self.initOptimizer()

    def initModel(self):
        models = []
        for i in range(self.args.site_number):
            models.append(ResNet18Model(num_classes=200))
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                for i in range(self.args.site_number):
                    models[i] = nn.DataParallel(models[i])
            for i in range(self.args.site_number):
                models[i] = models[i].to(self.device)
        return models

    def initOptimizer(self):
        optims = []
        for i in range(self.args.site_number):
            if self.args.layer is not None:
                optims.append(Adam(params=self.params_to_update[i], lr=self.args.lr))
            else:
                optims.append(Adam(params=self.models[i].parameters(), lr=self.args.lr))

        return optims

    def initDl(self):
        trn_dls = []
        for i in range(self.args.site_number):
            trn_dls.append(get_trn_loader(self.args.batch_size, site=i, device=self.device))

        val_dl = get_val_loader(self.args.batch_size, device=self.device)

        return trn_dls, val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            self.trn_writer = SummaryWriter(
                log_dir=self.logdir + '/trn-' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=self.logdir + '/val-' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        trn_dls, val_dl = self.initDl()

        val_best = 1e8
        validation_cadence = 5
        for epoch_ndx in range(1, self.args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.args.epochs,
                len(trn_dls[0]),
                len(val_dl),
                self.args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics = torch.zeros(5, len(trn_dls[0]), device=self.device)

            trnMetrics = self.doTraining(epoch_ndx, trn_dls)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics)

            self.mergeParams(names=None)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics, val_loss = self.doValidation(epoch_ndx, val_dl)
                self.logMetrics(epoch_ndx, 'val', valMetrics)
                val_best = min(val_loss, val_best)

                self.saveModel('mnist', epoch_ndx, val_loss == val_best)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dls):
        trnMetrics = torch.zeros(5, len(train_dls[0]), device=self.device)
        for i in range(self.args.site_number):

            self.models[i].train()

            log.warning('E{} Training on site {} ---/{} starting'.format(epoch_ndx, i,len(train_dls[i])))

            for batch_ndx, batch_tuple in enumerate(train_dls[i]):
                self.optims[i].zero_grad()

                loss = self.computeBatchLoss(
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

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.models[0].eval()
            valMetrics = torch.zeros(len(val_dl), device=self.device)

            log.warning('E{} Validation ---/{} starting'.format(epoch_ndx, len(val_dl)))

            for batch_ndx, batch_tuple in enumerate(val_dl):
                val_loss = self.computeBatchLoss(
                    batch_ndx,
                    batch_tuple,
                    valMetrics,
                    self.models[0],
                    'val'
                )
                if batch_ndx % 50 == 0:
                    log.info('E{} Validation {}/{}'.format(epoch_ndx, batch_ndx, len(val_dl)))

        return valMetrics.to('cpu'), val_loss

    def computeBatchLoss(self, batch_ndx, batch_tup, metrics, model, mode):
        batch, labels = batch_tup
        batch = batch.to(device=self.device, non_blocking=True)
        labels = labels.to(device=self.device, non_blocking=True)

        if mode == 'trn':
            angle = random.choice([0, 90, 180, 270])
            flip = random.choice([True, False])
            batch = functional.rotate(batch, angle)
            if flip:
                batch = functional.hflip(batch)

        pred = model(batch)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred, labels)

        metrics[batch_ndx] = torch.FloatTensor([
            loss.detach(),
        ])

        return loss.mean()

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics,
        img_list=None
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

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

    def mergeParams(self, names=None):
        dicts = []
        for i in range(self.args.site_number):
            dicts.append(self.models[i].state_dict())
        dict_avg = {}

        names = self.models[0].named_parameters()

        for name, _ in names:
            dict_avg[name] = 0
            for i in range(self.args.site_number):
                dict_avg[name] += dicts[i][name]
            dict_avg[name] = dict_avg[name] / self.args.site_number

        for i in range(self.args.site_number):
            self.models[i].load_state_dict(dict_avg, strict=False)


if __name__ == '__main__':
    MultiSiteTrainingApp().main()
