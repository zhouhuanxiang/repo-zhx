import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize


class ResNet(nn.Module):
    def __init__(self, use_last=False):
        super(ResNet, self).__init__()

        self.training = False
        self.use_last = use_last

        self.down_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )#256
        self.down_layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),#128
            BasicBlock(64, 128)
        )
        self.down_layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),#64
            BasicBlock(128, 192)
        )
        self.down_layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),#32
            BasicBlock(192, 256)
        )
        self.down_layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),#16
            BasicBlock(256, 320)
        )
        self.psp = PSPModule(320, 320)
        self.up_layer1 = up(320, 256)#32
        self.up_layer2 = up(256, 192)#64
        self.up_layer3 = up(192, 128)#128
        self.up_layer4 = up(128, 64)#256
        if self.use_last:
            self.out = outconv(64, 1)

        for module in self.children():
            module.eval()

    def train(self, mode):
        return self

    def forward(self, x):
        B, C, H, W = x.shape
        x = Normalize([0.485, 0.456, 0.406] * B, [0.229, 0.224, 0.225] * B)(x.view(B * C, H, W))
        x = x.view(B, C, H, W)

        x1 = self.down_layer1(x) #64
        x2 = self.down_layer2(x1) #128
        x3 = self.down_layer3(x2) #256
        x4 = self.down_layer4(x3) #392
        x5 = self.down_layer5(x4) #512
        x6 = self.psp(x5)
        x = self.up_layer1(x6, x4) #512+392
        x = self.up_layer2(x, x3) #392+256
        x = self.up_layer3(x, x2) #256+128
        x = self.up_layer4(x, x1) #128+64

        if self.use_last:
            x = self.out(x)
            return F.sigmoid(x)
        else:
            return x

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes)
        )
        self.stride = stride

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = BasicBlock(out_ch+in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

if __name__ == '__main__':
    x = torch.randn((2, 3, 256, 256))
    mask = torch.randn((2, 1, 256, 256))
    net = UNet_Res()
    print(net)
    out = net(x)
    print(1)