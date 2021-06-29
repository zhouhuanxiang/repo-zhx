'''
https://github.com/yuanjunchai/IKC
https://bbs.cvmart.net/topics/2048


'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.deblur_archs.ResNet_4 as ResNet_4

# blurnet_nf = 1
udvd_nf = 32
udvd_nf_up = 32 * 4
ndvd_nb = 8

class UDVDPlus(nn.Module):
    def __init__(self, blurnet_nf):
        super().__init__()
        self.uvud = UDVD(blurnet_nf)
        if blurnet_nf == 64:
            self.resnet = ResNet_4.ResNet(use_last=False).eval()
        elif blurnet_nf == 1:
            self.resnet = ResNet_4.ResNet(use_last=True).eval()

    def forward(self, x):
        fea = self.resnet(x)

        # B, C, H, W = x.shape
        # fea = torch.zeros(B, blurnet_nf, H, W).to(x.device)

        y = self.uvud(x, fea)

        return y

class UDVD(nn.Module):
    def __init__(self, blurnet_nf):
        super().__init__()
        self.blurnet_nf = blurnet_nf
        self.head = nn.Conv2d(3 + blurnet_nf, udvd_nf, 3, 1, 1)
        body = [ResBlock(udvd_nf, 3, 0.1) for _ in range(ndvd_nb)]
        self.body = nn.Sequential(*body)
        self.UpDyConv = UpDynamicConv()
        self.ComDyConv1 = CommonDynamicConv()
        # self.ComDyConv2 = CommonDynamicConv()

    def forward(self, image, kernel):
        assert image.size(1) == 3, 'Channels of Image should be 3, not {}'.format(image.size(1))
        assert kernel.size(1) == self.blurnet_nf, 'Channels of kernel should be 15, not {}'.format(kernel.size(1))
        inputs = torch.cat([image, kernel], 1)
        head = self.head(inputs)
        body = self.body(head) + head
        # output1 = self.UpDyConv(image, body)
        output2 = self.ComDyConv1(image, body)
        # output3 = self.ComDyConv2(output2, body)
        return output2

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, res_scale=1.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, 1, padding)
        )
        self.res_scale = res_scale

    def forward(self, inputs):
        return inputs + self.conv(inputs) * self.res_scale

class PixelConv(nn.Module):
    def __init__(self, scale=2, depthwise=False):
        super().__init__()
        self.scale = scale
        self.depthwise = depthwise

    def forward(self, feature, kernel):
        NF, CF, HF, WF = feature.size()
        NK, ksize, HK, WK = kernel.size()
        assert NF == NK and HF == HK and WF == WK
        if self.depthwise:
            ink = CF
            outk = 1
            ksize = int(np.sqrt(int(ksize // (self.scale ** 2))))
            pad = (ksize - 1) // 2
        else:
            ink = 1
            outk = CF
            ksize = int(np.sqrt(int(ksize // CF // (self.scale ** 2))))
            pad = (ksize - 1) // 2

        # features unfold and reshape, same as PixelConv
        feat = F.pad(feature, [pad, pad, pad, pad])
        feat = feat.unfold(2, ksize, 1).unfold(3, ksize, 1)
        feat = feat.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat = feat.reshape(NF, HF, WF, ink, -1)

        # kernel
        kernel = kernel.permute(0, 2, 3, 1).reshape(NK, HK, WK, ksize * ksize, self.scale ** 2 * outk)

        output = torch.matmul(feat, kernel)
        output = output.permute(0, 3, 4, 1, 2).view(NK, -1, HF, WF)
        if self.scale > 1:
            output = F.pixel_shuffle(output, self.scale)
        return output

class CommonDynamicConv(nn.Module):
    def __init__(self):
        super().__init__()
        udvd_nf_half = udvd_nf // 2
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, udvd_nf_half, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(udvd_nf_half, udvd_nf_half, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(udvd_nf_half, udvd_nf, 3, 1, 1)
        )
        # I'm not sure how to deal the feautre.
        # Because it need to upsample the feature and align,
        # but the paper not provide useful information about it, just provide
        # Sub-pixel Convolution layer is used to align the resolutions between paths.
        self.feat_conv = nn.Sequential(
            # nn.PixelShuffle(2),
            # nn.Conv2d(32, 128, 1)
            nn.Conv2d(udvd_nf, udvd_nf_up, 1)
        )
        self.feat_residual = nn.Sequential(
            nn.Conv2d(udvd_nf + udvd_nf_up, udvd_nf_half, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(udvd_nf_half, 3, 3, 1, 1)
        )
        self.feat_kernel = nn.Conv2d(udvd_nf + udvd_nf_up, 25, 3, 1, 1)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

    def forward(self, image, features):
        image_conv = self.image_conv(image)
        features = self.feat_conv(features)
        cat_inputs = torch.cat([image_conv, features], 1)

        kernel = self.feat_kernel(cat_inputs)
        # print(kernel.shape)
        output = self.pixel_conv(image, kernel)

        residual = self.feat_residual(cat_inputs)
        return output + residual

class UpDynamicConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1)
        )
        self.feat_residual = nn.Sequential(
            nn.Conv2d(160, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(16, 3, 3, 1, 1)
        )
        self.feat_kernel = nn.Conv2d(160, 25 * 4, 3, 1, 1)
        self.pixel_conv = PixelConv(scale=2, depthwise=True)

    def forward(self, image, features):
        image_conv = self.image_conv(image)
        cat_inputs = torch.cat([image_conv, features], 1)

        kernel = self.feat_kernel(cat_inputs)
        output = self.pixel_conv(image, kernel)

        residual = self.feat_residual(cat_inputs)
        return output + residual

def demo():
    net = UDVD()

    inputs = torch.randn(1, 3, 64, 64)
    kernel = torch.randn(1, 15, 64, 64)
    noise = torch.randn(1, 1, 64, 64)

    with torch.no_grad():
        output1, output2, output3 = net(inputs, kernel, noise)

    print(output1.size())
    print(output2.size())
    print(output3.size())

if __name__ == '__main__':
    demo()