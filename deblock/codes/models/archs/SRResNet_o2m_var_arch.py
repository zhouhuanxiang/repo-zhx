import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util


class ResidualBlock_noBN_withZ(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, ni=65, no=64):
        super(ResidualBlock_noBN_withZ, self).__init__()
        self.conv1 = nn.Conv2d(ni, ni, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(ni, no, 3, 1, 1, bias=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity[:, :out.shape[1], :, :] + out

class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale
        self.nb = nb

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        # basic_block = functools.partial(ResidualBlock_noBN_withZ, nf=nf)
        # self.recon_trunk = arch_util.make_layer(basic_block, nb)
        self.recon_trunk = nn.ModuleList([ResidualBlock_noBN_withZ(nf + 1, nf) for i in range(nb)])
        # upsampling
        self.upconv1 = nn.Conv2d(nf + 1, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf + 1, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)

    def forward(self, x, var):

        z = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        # z = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        # z = z * var
        
        out = self.lrelu(self.conv_first(x))
        # out = self.recon_trunk(fea)
        for i, layer in enumerate(self.recon_trunk):
            out = layer(torch.cat((out, z), dim=1))

        out = self.lrelu(self.upconv1(torch.cat((out, z), dim=1)))
        out = self.conv_last(self.lrelu(self.HRconv(torch.cat((out, z), dim=1))))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        if out.shape[1] == base.shape[1]:
            out += base
        else:
            out += base[:, :3, :, :]
        return out
