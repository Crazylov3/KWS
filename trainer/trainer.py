import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from utils.generals import load_checkpoint, save_checkpoint, AverageMeter, accuracy, increment_path, save_strategy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import wandb


class KWSTrainer:
    def __init__(
            self,
            model,
            optimizer,
            criterion,
            lr_scheduler,
            train_loader,
            test_loader,
            device,
            n_epoch,
            pretrained_path="",
            output_dir="",
            distributed=False,
            warmup_epoch=5,
            warmup_lr=None,
            use_wandb=False,
            use_tensorboard=False,
            config=None,
            max_norm=0,
            save_epoch=5,
            exist_ok=False,
    ):
        if warmup_lr is None:
            warmup_lr = [0, 0.1]
        assert distributed is False, "Not supported yet!"
        if use_wandb:
            assert config, "Please specify training config"
        self.model = nn.DataParallel(model).to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.total_epochs = n_epoch
        self.warmup_epoch = warmup_epoch
        self.optimizer = optimizer
        self.lr_schedular = lr_scheduler
        self.pretrained_path = pretrained_path
        self.max_norm = max_norm
        self.warmup_lr = warmup_lr
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.save_epoch = save_epoch
        self.save_root = increment_path(output_dir, exist_ok=exist_ok)

        if use_tensorboard:
            self.writer = SummaryWriter(self.save_root)

        if use_wandb:
            self.wandb_run = wandb.init(
                project="KWS_v2",
                name=self.save_root.split("/")[-1],
                resume="allow",
                config=config
            )

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        loss_metric = AverageMeter()
        acc_metric = AverageMeter()
        pdar = tqdm(self.train_loader, desc=f"Epoch: {epoch}/{self.total_epochs}")
        for spec, label in pdar:
            spec = spec.to(self.device)
            label = label.to(self.device)

            oup = self.model(spec)
            loss = self.criterion(oup, label)
            acc, _ = accuracy(oup, label)

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            loss_metric.update(loss.item())
            acc_metric.update(acc)

            pdar.set_postfix({
                "train loss": loss_metric.avg,
                "train accuracy": acc_metric.avg
            })
        return loss_metric.avg, acc_metric.avg

    # @torch.no_grad()
    def _evaluate(self):
        loss, acc, _ = self.evaluate(self.model, self.criterion, self.test_loader, self.device)
        return loss, acc

    @staticmethod
    @torch.no_grad()
    def evaluate(model, criterion, loader, device):
        model.eval()
        criterion.eval()
        confusion = np.zeros((2, 2), dtype="int32")
        loss_metric = AverageMeter()
        acc_metric = AverageMeter()
        pdar = tqdm(loader, desc=f"Evaluating...")
        for spec, label in pdar:
            spec = spec.to(device)
            label = label.to(device)

            oup = model(spec)

            loss = criterion(oup, label)

            acc, conf = accuracy(oup, label)
            confusion += conf

            loss_metric.update(loss.item())
            acc_metric.update(acc)
            pdar.set_postfix({
                "test loss": loss_metric.avg,
                "accuracy": acc_metric.avg
            })
        return loss_metric.avg, acc_metric.avg, confusion

    def train(self):
        if Path(self.pretrained_path).is_file():
            self.model, self.optimizer, self.lr_schedular, start_epoch, best_loss, best_acc = load_checkpoint(
                self.model,
                self.optimizer,
                self.lr_schedular,
                self.pretrained_path
            )
        else:
            start_epoch = 0
            best_loss = 1e9
            best_acc = 0

        for epoch in range(start_epoch, self.total_epochs):
            if epoch < self.warmup_epoch:
                for g in self.optimizer.param_groups:
                    g["lr"] = np.interp(epoch, [0, self.warmup_epoch - 1], self.warmup_lr)  # [start, end]
            train_loss, train_acc = self._train_one_epoch(epoch)
            test_loss, test_acc = self._evaluate()

            if epoch >= self.warmup_epoch:
                self.lr_schedular.step()

            if self.use_tensorboard:
                tags = ["train/loss", "train/accuracy", "test/loss", "test/accuracy", "lr"]
                items = [train_loss, train_acc, test_loss, test_acc, self.optimizer.param_groups[0]['lr']]
                for tag, item in zip(tags, items):
                    self.writer.add_scalar(tag, item, epoch)

            if self.use_wandb:
                self.wandb_run.log({
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "test/loss": test_loss,
                    "test/accuracy": test_acc,
                    "lr": self.optimizer.param_groups[0]['lr']
                })

            if epoch > self.save_epoch:
                if save_strategy(test_loss, test_acc, best_loss, best_acc):
                    best_loss = test_loss
                    best_acc = test_acc
                    save_checkpoint(self.model, self.optimizer, self.lr_schedular, epoch, test_loss, test_acc,
                                    self.save_root + "/checkpoint_best.pth.taz")
                else:
                    save_checkpoint(self.model, self.optimizer, self.lr_schedular, epoch, test_loss, test_acc,
                                    self.save_root + "/checkpoint_last.pth.taz")

        if self.use_wandb:
            wandb.finish()
