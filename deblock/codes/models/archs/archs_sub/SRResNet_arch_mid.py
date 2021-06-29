import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk_1 = arch_util.make_layer(basic_block, nb // 2)
        self.recon_trunk_2 = arch_util.make_layer(basic_block, nb // 2)
        # upsampling
        # self.upconv1_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last_1 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.upconv1_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last_2 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1_2, self.HRconv_2, self.conv_last_2, self.HRconv_1, self.conv_last_1],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out1 = self.recon_trunk_1(fea)
        out2 = self.recon_trunk_2(out1)

        # out1 = self.lrelu(self.upconv1_1(out1))
        out1 = self.conv_last_1(self.lrelu(self.HRconv_1(out1)))

        out2 = self.lrelu(self.upconv1_2(out2))
        out2 = self.conv_last_2(self.lrelu(self.HRconv_2(out2)))

        out1 += x
        out2 += x
        return out1, out2
