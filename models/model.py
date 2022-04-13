import torch
from torch import nn
from models.modules import TransitionBlock, NormalBlock


class BcResNetModel(nn.Module):
    def __init__(self, n_class: int = 35, *, scale: 1, dropout: float = 0.1, use_subspectral: bool = True):
        super().__init__()

        self.input_conv = nn.Conv2d(1, int(16 * scale), kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock(int(16 * scale), int(8 * scale), dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock(int(8 * scale), dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlock(int(8 * scale), int(12 * scale), dilation=2, stride=2, dropout=dropout,
                                  use_subspectral=use_subspectral)
        self.n21 = NormalBlock(int(12 * scale), dilation=2, dropout=dropout, use_subspectral=use_subspectral)

        self.t3 = TransitionBlock(int(12 * scale), int(16 * scale), dilation=4, stride=2, dropout=dropout,
                                  use_subspectral=use_subspectral)
        self.n31 = NormalBlock(int(16 * scale), dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock(int(16 * scale), dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock(int(16 * scale), dilation=4, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlock(int(16 * scale), int(20 * scale), dilation=8, dropout=dropout,
                                  use_subspectral=use_subspectral)
        self.n41 = NormalBlock(int(20 * scale), dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock(int(20 * scale), dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock(int(20 * scale), dilation=8, dropout=dropout, use_subspectral=use_subspectral)

        self.dw_conv = nn.Conv2d(int(20 * scale), int(20 * scale), kernel_size=(5, 5), groups=int(20 * scale))
        self.onexone_conv = nn.Conv2d(int(20 * scale), int(32 * scale), kernel_size=1)

        self.head_conv = nn.Conv2d(int(32 * scale), n_class, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=3, keepdim=True)
        x = self.head_conv(x)

        x = x.squeeze(-1).squeeze(-1)
        return x


if __name__ == "__main__":
    from thop import profile

    model = BcResNetModel(n_class=2, scale=1.5, dropout=0.1).cuda()
    x = torch.ones(2, 1, 40, 151).cuda()
    oup = model(x)
    print(oup.shape)

    ops, param = profile(model, inputs=(x,), verbose=False)
    print("ops: ", ops)
    print("param: ", param)
