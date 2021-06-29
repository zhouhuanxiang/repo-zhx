import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util


def PixelUnshuffle(fm, r):
    b, c, h, w = fm.shape
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    fm_view = fm.contiguous().view(b, c, out_h, r, out_w, r)
    fm_prime = fm_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)
    return fm_prime

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, downsample_rate):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.downsample_rate = downsample_rate

        self.dim_conv_1 = nn.Conv2d(in_channels=in_dim * (downsample_rate ** 2), out_channels=in_dim, kernel_size=3, stride=1, padding=1)
        self.dim_conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim * (downsample_rate ** 2), kernel_size=3, stride=1, padding=1)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, xx):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = PixelUnshuffle(xx, self.downsample_rate)
        x = self.dim_conv_1(x)

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        # return out, attention
        out = self.dim_conv_2(out)
        out = F.pixel_shuffle(out, self.downsample_rate)

        out = self.gamma * out + xx
        return out

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
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

        self.attn1 = Self_Attn(nf, 'relu', 4)

    def forward(self, x):
        out = self.lrelu(self.conv_first(x))
        out = self.recon_trunk_1(out)

        out = self.attn1(out)

        out = self.recon_trunk_2(out)

        out = self.lrelu(self.upconv1(out))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        if out.shape[1] == base.shape[1]:
            out += base
        else:
            out += base[:, :3, :, :]
        return out
