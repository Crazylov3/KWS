import torchaudio
from data_loader.dataset import KWSDataset
from models.model import BcResNetModel
from utils.generals import Criterion
from trainer.trainer import KWSTrainer
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    model = BcResNetModel(n_class=2, scale=1.5, dropout=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.DataParallel(model).to(device)
    ckpt = torch.load(r"D:\LTGiang\Intership\RikkeiAI\BC_ResNet_from_scratch\runs\KWSexp2\checkpoint_best.pth.taz",
                      map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = KWSDataset("data/train_sample_v2", "data/test_csv_final.csv")
    loader = DataLoader(dataset, shuffle=False, batch_size=100)

    criterion = Criterion()

    _, _, conf = KWSTrainer.evaluate(model, criterion, loader, device)
    print(conf)
