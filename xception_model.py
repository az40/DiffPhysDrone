import torch
import torch.nn.functional as F
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        # if strides != 1:
        #     rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0., global_pool='avg'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = 2048

        self.conv1 = nn.Conv2d(in_chans, 32, 2, 2, 0, bias=False)  # 1, 12, 16 -> 32, 6, 8
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)  #  32, 6, 8 -> 64, 4, 6
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, 2, 1, start_with_relu=False)  # 64, 4, 6 -> 128, 4, 6
        self.block2 = Block(128, 256, 2, 1)  # 128, 4, 6 -> 256, 4, 6
        self.block3 = Block(256, 728, 2, 1)  # 256, 4, 6 -> 728, 4, 6

        self.block4 = Block(728, 728, 3, 1)  # 728, 4, 6 -> 728, 4, 6
        self.block5 = Block(728, 728, 3, 1)  # 728, 4, 6 -> 728, 4, 6
        self.block6 = Block(728, 728, 3, 1)  # 728, 4, 6 -> 728, 4, 6
        self.block7 = Block(728, 728, 3, 1)  # 728, 4, 6 -> 728, 4, 6

        self.block8 = Block(728, 728, 3, 1)  # 728, 4, 6 -> 728, 4, 6
        self.block9 = Block(728, 728, 3, 1)  # 728, 4, 6 -> 728, 4, 6
        self.block10 = Block(728, 728, 3, 1)  # 728, 4, 6 -> 728, 4, 6
        self.block11 = Block(728, 728, 3, 1)  # 728, 4, 6 -> 728, 4, 6

        self.block12 = Block(728, 1024, 2, 1, grow_first=False)  # 728, 4, 6 -> 1024, 4, 6

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)  # 1024, 4, 6 -> 1536, 4, 6
        self.bn3 = nn.BatchNorm2d(1536)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536, self.num_features, 3, 1, 1)  # 1536, 4, 6 -> 2048, 4, 6
        self.bn4 = nn.BatchNorm2d(self.num_features)
        self.act4 = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.AvgPool2d(4, 6),
            nn.Flatten(),
            nn.Linear(2048, 192, bias=False),
        )
        self.feature_info = [
            dict(num_chs=64, reduction=2, module='act2'),
            dict(num_chs=128, reduction=4, module='block2.rep.0'),
            dict(num_chs=256, reduction=8, module='block3.rep.0'),
            dict(num_chs=728, reduction=16, module='block12.rep.0'),
            dict(num_chs=2048, reduction=32, module='act4'),
        ]

        # #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)    
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        # x = self.global_pool(x)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



class XceptionModel(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4, in_channels=1) -> None:
        super().__init__()
        self.stem = Xception(in_chans=1)
        self.v_proj = nn.Linear(dim_obs, 192)
        # self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        # self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, None, hx


if __name__ == '__main__':
    x = torch.randn(64, 1, 12, 16)
    model = XceptionModel()
    print(model)
