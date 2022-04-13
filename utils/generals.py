import torch
import torch.nn as nn
from pathlib import Path
import re
import glob
from data_loader.dataset import KWSDataset
from sklearn.metrics import confusion_matrix


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Criterion(nn.Module):
    # Just CrossEntropyLoss
    def __init__(self):
        super(Criterion, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        return loss


def accuracy(pred, target):
    class_indx = torch.argmax(pred, dim=1)
    assert class_indx.shape == target.shape
    correct = class_indx == target
    target, class_indx = target.detach().cpu().numpy(), class_indx.detach().cpu().numpy()
    return sum(correct).item() / len(target), confusion_matrix(target, class_indx)


def load_checkpoint(model, optimizer, lr_schedular, pretrain_path):
    ckpt = torch.load(pretrain_path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_schedular.load_state_dict(ckpt["lr_schedular"])
    epoch = ckpt["epoch"] + 1
    loss = ckpt["loss"]
    acc = ckpt["accuracy"]
    return model, optimizer, lr_schedular, epoch, loss, acc


def save_checkpoint(model, optimizer, lr_schedular, epoch, loss, acc, checkpoint_path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_schedular": lr_schedular.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "accuracy": acc
    }, checkpoint_path)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"


def save_strategy(current_loss, current_acc, best_loss, best_acc):
    return current_acc > best_acc


def convet2logmel(wave_form, sample_rate):
    return KWSDataset.prepare_wav(wave_form, sample_rate, sample_rate)


@torch.no_grad()
def trigger_word_detect(model, _spectrum, device):
    assert _spectrum.shape == (1, 1, 40, 151)
    _spectrum = _spectrum.to(device)
    oup = model(_spectrum)
    oup = torch.softmax(oup, dim=1)
    pred = torch.argmax(oup, dim=1)[0].item()
    _prob = oup[0][1].item()
    return pred == 1, _prob
