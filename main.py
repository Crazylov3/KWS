import torch
from torch.optim import lr_scheduler, SGD
from models.model import BcResNetModel
from utils.generals import Criterion
from data_loader.dataset import KWSDataset
from torch.utils.data import DataLoader
from trainer.trainer import KWSTrainer
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--number-classes", type=int, default=2)
    parser.add_argument("--model-scale", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup-epoch", type=int, default=5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--root-file", type=str, default="data/train_sample_v2")
    parser.add_argument("--train-txt", type=str, default="data/train_csv_final.csv")
    parser.add_argument("--test-txt", type=str, default="data/test_csv_final.csv")
    parser.add_argument("--pretrain-path", type=str, default="runs/KWSexp2/checkpoint_last.pth.taz")
    parser.add_argument("--save-epoch", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs/KWSexp")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--device", type=str, default="2")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    device = config.__dict__.pop('device')
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BcResNetModel(n_class=config.number_classes, scale=config.model_scale, dropout=config.dropout)
    criterion = Criterion()
    optim = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay,
                nesterov=True)
    _lr_scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=config.epochs)

    trainset = KWSDataset(config.root_file, csv_file=config.train_txt)
    testset = KWSDataset(config.root_file, csv_file=config.test_txt)

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    config.lr_scheduler = f"{_lr_scheduler.__class__.__name__}"
    config.optimizer = f"{optim.__class__.__name__}"
    use_wandb = config.__dict__.pop('use_wandb')
    use_tensorboard = config.__dict__.pop('use_tensorboard')
    output_dir = config.__dict__.pop('output_dir')
    pretrain_path = config.__dict__.pop("pretrain_path")
    exist_ok = config.__dict__.pop("exist_ok")
    delattr(config, 'root_file')
    delattr(config, 'train_txt')
    delattr(config, 'test_txt')

    train_object = KWSTrainer(
        model=model,
        optimizer=optim,
        criterion=criterion,
        lr_scheduler=_lr_scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        pretrained_path=pretrain_path,
        device=device,
        n_epoch=config.epochs,
        output_dir=output_dir,
        warmup_epoch=config.warmup_epoch,
        warmup_lr=[0, config.lr],
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        save_epoch=config.save_epoch,
        config=config,
        exist_ok=exist_ok,
    )

    train_object.train()
