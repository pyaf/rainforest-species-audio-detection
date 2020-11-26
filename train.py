import warnings
import os
import pdb
import time
from datetime import datetime
import _thread

from apex import amp
import pandas as pd
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import defaultdict
from tqdm import tqdm

# from ssd import build_ssd
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import Logger
from shutil import copyfile

from models import get_model
from utils import *
from optimizers import get_optimizer
from losses import get_loss
from datasets import provider
from pathlib import Path

warnings.filterwarnings("ignore")
HOME = os.path.abspath(os.path.dirname(__file__))
now = datetime.now()
date = "%s-%s" % (now.day, now.month)
# print(HOME)


class Trainer(object):
    def __init__(self):
        # seed_pytorch()
        self.args = get_parser()
        self.cfg = load_cfg(self.args)
        self.model_name = self.cfg.model_name
        self.filename = Path(self.args.filepath).stem
        # {date}_{self.model_name}_f{self.fold}_{ext_text}
        self.folder = Path(f"weights/{self.filename}")
        self.cfg["folder"] = self.folder
        # self.resume = self.cfg['resume']
        self.resume = self.args.resume
        self.batch_size = self.cfg["batch_size"]
        self.accumulation_steps = {x: 32 // bs for x, bs in self.batch_size.items()}
        self.num_classes = self.cfg["num_classes"]
        self.ep2unfreeze = self.cfg["ep2unfreeze"]
        self.num_epochs = self.cfg["num_epochs"]
        self.base_lr = self.cfg["base_lr"]
        self.momentum = self.cfg["momentum"]
        self.patience = self.cfg["patience"]
        self.phases = self.cfg["phases"]
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.cuda = torch.cuda.is_available()
        torch.set_num_threads(12)
        self.device = torch.device("cuda" if self.cuda else "cpu")
        # self.df_path = self.cfg['df_path']
        self.resume_path = os.path.join(HOME, self.folder, "ckpt.pth")
        # self.resume_path = self.cfg['resume_path']
        self.save_folder = os.path.join(HOME, self.folder)
        self.model_path = os.path.join(self.save_folder, "model.pth")
        self.ckpt_path = os.path.join(self.save_folder, "ckpt.pth")
        self.net = get_model(self.model_name, self.num_classes)
        self.criterion = get_loss(self.cfg)
        self.optimizer = get_optimizer(self.cfg, self.net.parameters())
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=self.patience, verbose=True
        )
        logger = logger_init(self.save_folder)
        self.log = logger.info
        if self.resume:
            self.load_state()
        else:
            self.initialize_net()
        self.net = self.net.to(self.device)

        # Mixed precision training
        self.net, self.optimizer = amp.initialize(
            self.net, self.optimizer, opt_level="O1", verbosity=0
        )
        if self.cuda:
            cudnn.benchmark = True
        self.tb = {
            x: Logger(os.path.join(self.save_folder, "logs", x)) for x in self.phases
        }  # tensorboard logger, see [3]
        mkdir(self.save_folder)
        df = pd.read_feather(str(self.cfg.home / self.cfg.df_path))
        self.dataloaders = {phase: provider(df, phase, self.cfg) for phase in self.phases}
        #check_sanctity(self.dataloaders)
        save_cfg(self.cfg, self)

    def load_state(self):  # [4]
        if self.resume:
            path = self.resume_path
            self.log("Resuming training, loading {} ...".format(path))
        state = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state["state_dict"])

        if self.resume:
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_loss = state["best_loss"]
            self.start_epoch = state["epoch"] + 1
            if self.start_epoch >= self.cfg['ep2unfreeze']:
                for params in self.net.parameters():
                    params.requires_grad = True
                print('All parameters are trainable')

        if self.cuda:
            for opt_state in self.optimizer.state.values():
                for k, v in opt_state.items():
                    if torch.is_tensor(v):
                        opt_state[k] = v.to(self.device)

    def initialize_net(self):
        # using `pretrainedmodels` library, models are already pretrained
        pass

    def forward(self, images, targets):
        #pdb.set_trace()
        images = images.to(self.device)
        targets = targets.type(torch.FloatTensor).to(self.device) # [1]
        outputs = self.net(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def iterate(self, epoch, phase):
        start = time.strftime("%H:%M:%S")
        self.log(f"Starting epoch: {epoch} | phase: {phase} | {start}")
        meter = Meter(phase, epoch, self.save_folder)
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        accu_steps = self.accumulation_steps[phase]
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            fnames, images, targets = batch
            #pdb.set_trace()
            #targets = torch.stack(targets, axis=1)
            loss, outputs = self.forward(images, targets)
            loss = loss / accu_steps
            if phase == "train":
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if (itr + 1) % accu_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(targets, outputs.detach())
            tk0.set_postfix(loss=((running_loss * accu_steps) / ((itr + 1))))
        epoch_loss = (running_loss * accu_steps) / total_batches
        acc = epoch_log(
            self.optimizer, self.log, self.tb, phase, epoch, epoch_loss, meter, start
        )
        torch.cuda.empty_cache()
        return epoch_loss, acc

    def train(self):
        t0 = time.time()
        if self.start_epoch >= self.ep2unfreeze:
            for params in self.net.parameters():
                params.requires_grad = True

        #self.optimizer = adjust_lr(1e-4, self.optimizer)
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            train_loss, acc = self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state, self.ckpt_path)  # [2]
            self.scheduler.step(train_loss)

            continue
            if "val_new" in self.phases:
                self.iterate(epoch, "val_new")
            with torch.no_grad():
                val_loss, val_acc = self.iterate(epoch, "val")
                #torch.save(state, self.ckpt_path)  # [2]
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.log("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, self.model_path)
            copyfile(
                self.ckpt_path, os.path.join(self.save_folder, "ckpt%d.pth" % epoch)
            )
            if epoch == 0 and len(self.dataloaders["train"]) > 100:
                # make sure train/val ran error free, and it's not debugging
                #commit(self.filename)
                pass
            # print_time(self.log, t_epoch_start, "Time taken by the epoch")
            print_time(self.log, t0, "Total time taken so far")
            print()
            """ progressive resizing
            if (epoch+1) % 5 == 0:
                self.cfg['size'] = [299, 384, 512, 784, 1024][(epoch+1) // 5]
                if self.cfg['size'] == 512:
                    self.cfg['batch_size'] = {'train': 4, 'val':4}
                elif self.cfg['size'] > 512:
                    self.cfg['batch_size'] = {'train': 2, 'val':2}
                self.log('*** Setting size to %d ***' % self.cfg['size'])
                self.log('batch size %s' % self.cfg['batch_size'])
                self.dataloaders = {
                    phase: provider(phase, self.cfg) for phase in self.phases
                }
            """
            # self.log("\n" + "=" * 60 + "\n")
            print()


if __name__ == "__main__":
    model_trainer = Trainer()
    model_trainer.train()

"""Footnotes
[1]: Crossentropy loss functions expects targets to be in labels (not one-hot) and of type
LongTensor, BCELoss expects targets to be FloatTensor

[2]: the ckpt.pth is saved after each train and val phase, val phase is neccessary becausue we want the best_threshold to be computed on the val set., Don't worry, the probability of your system going down just after a crucial training phase is low, just wait a few minutes for the val phase :p

[3]: one tensorboard logger for train and val each, in same folder, so that we can see plots on the same graph

[4]: if pretrained is true, a model state from self.pretrained path will be loaded, if self.resume is true, self.resume_path will be loaded, both are true, self.resume_path will be loaded
"""
