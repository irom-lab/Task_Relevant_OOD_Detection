import torch.nn as nn
from models.layers import StochasticLinear as SLinear
from models.layers import StochasticConv2d as SConv2d
from models.layers import NotStochasticLinear as Linear
from models.layers import NotStochasticConv2d as Conv2d
from models.layers import StochasticModel


class NSPolicy(StochasticModel):
    def __init__(self, linear=Linear, conv=Conv2d):
        super().__init__()
        # 50x50 input
        n_filt = 64
        output_size = 7
        self.conv1 = conv((n_filt, 1, 4, 4), 3, 0)
        self.conv2 = conv((n_filt, n_filt, 3, 3), 2, 0)
        self.conv3 = conv((n_filt, n_filt, 3, 3), 1, 0)
        self.conv4 = conv((n_filt, n_filt, 3, 3), 1, 0)
        self.lin1 = linear(n_filt, n_filt)
        self.lin2 = linear(n_filt, output_size)
        self.act = nn.ReLU()
        self.sm = nn.Softmax()
        self.bn = nn.BatchNorm2d(n_filt)

    def forward(self, x):
        x = self.act(self.bn(self.conv1(x)))
        x = self.act(self.bn(self.conv2(x)))
        x = self.act(self.bn(self.conv3(x)))
        x = self.act(self.bn(self.conv4(x)))
        x = x.mean(dim=[2, 3])
        x = self.act(self.lin1(x))
        x = self.sm(self.lin2(x))
        return x


class SPolicy(NSPolicy):
    def __init__(self):
        super().__init__(linear=SLinear, conv=SConv2d)

