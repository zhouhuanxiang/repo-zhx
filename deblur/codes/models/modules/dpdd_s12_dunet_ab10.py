import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv

import pytorch_wavelets

'''
加入动态卷积
'''


# ==========
# Spatio-temporal deformable fusion module
# ==========

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class DWTForward(nn.Module):
    def __init__(self, nf=64):
        super(DWTForward, self).__init__()
        self.nf = nf
        self.DWTForward = pytorch_wavelets.DWTForward()

    def forward(self, x):
        yl, yh = self.DWTForward(x.view(-1, x.shape[2], x.shape[3], x.shape[4]))

        yh = yh[0]
        # print('yh', yh.shape)
        N, C, _, H, W = yh.shape

        yh = yh.view(N, C * 3, H, W)
        y = torch.cat((yl, yh), dim=1)
        y = y.view(x.shape[0], x.shape[1], y.shape[1], y.shape[2], y.shape[3])

        return y


class DWTInverse(nn.Module):
    def __init__(self, nf=64):
        super(DWTInverse, self).__init__()
        self.nf = nf
        self.DWTInverse = pytorch_wavelets.DWTInverse()

    def forward(self, x):
        yl = x[:, :x.shape[1] // 4, :, :]
        yh = x[:, x.shape[1] // 4:, :, :]

        N, C, H, W = yh.shape
        yh = [yh.view(N, C // 3, 3, H, W)]

        y = self.DWTInverse((yl, yh))
        return y


'''

import torch
import torch.nn.functional as F

a = torch.tensor(list(range(25)))
a = a.reshape(1,1,5,5).float()

bb = torch.nn.functional.unfold(a, 3, dilation=1, padding=1, stride=1).permute(0,2,1).reshape(1, 1, 5, 5, 3, 3)
torch.nn.functional.unfold(a, 3, dilation=2, padding=2, stride=1).permute(0,2,1).reshape(1, 1, 5, 5, 3, 3)

aa = F.pad(a, [1, 1, 1, 1])
aa = aa.unfold(2, 3, 1).unfold(3, 3, 1)

'''


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


class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv_1{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                )
            )
            setattr(
                self, 'dn_conv_dwt{}'.format(i), nn.Sequential(
                    nn.Conv2d(3 * 3, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    ResidualBlock_noBN(nf),
                )
            )
            setattr(
                self, 'dn_conv_concat{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf * 2, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                )
            )
            setattr(
                self, 'dn_conv_2{}'.format(i), nn.Sequential(
                    ResidualBlock_noBN(nf),
                    ResidualBlock_noBN(nf),
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    ResidualBlock_noBN(nf),
                    ResidualBlock_noBN(nf),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            ResidualBlock_noBN(nf),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc + 3, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

        self.kernel = nn.Conv2d(nf, 9, base_ks, padding=base_ks // 2)
        self.pixel_conv = PixelConv(scale=1, depthwise=True)

    def forward(self, inputs, inputs_l, inputs_h):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv_1 = getattr(self, 'dn_conv_1{}'.format(i))
            dn_conv_dwt = getattr(self, 'dn_conv_dwt{}'.format(i))
            dn_conv_concat = getattr(self, 'dn_conv_concat{}'.format(i))
            dn_conv_2 = getattr(self, 'dn_conv_2{}'.format(i))

            fea = dn_conv_1(out_lst[i - 1])
            dwt_fea = dn_conv_dwt(inputs_h[i - 1])
            concat_fea = dn_conv_concat(torch.cat([fea, dwt_fea], dim=1))
            out_lst.append(dn_conv_2(concat_fea))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        out = self.out_conv(out)
        off_msk = self.offset_mask(out)
        off = off_msk[:, :in_nc * 2 * n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc * 2 * n_off_msk:, ...]
        )

        # perform deformable convolutional fusion
        combine = self.deform_conv(inputs, off, msk)
        dcn_feat = F.relu(
            combine[:, 3:, ::],
            inplace=False
        )
        recon = combine[:, :3, ::]

        kernel = self.kernel(out)
        dyn_feat = self.pixel_conv(inputs, kernel)
        fused_feat = torch.cat((dcn_feat, dyn_feat), dim=1)

        # offset均值
        off_mean = torch.mean(torch.abs(off), dim=1, keepdim=True)
        return fused_feat, -1, [off_mean, 0]

        # off_var = off_mean
        # off_var = off - off_mean.repeat(1, 108, 1, 1)
        # off_var = torch.pow(torch.abs(off_var), 2)
        # off_var = torch.sqrt(torch.mean(off_var, dim=1, keepdim=True))

        # print('{},'.format(torch.mean(torch.abs(off)).item(), ))
        # print(off[0, :, 197, 1413])

        # dcn_feat = dcn_feat[:, :, 20:-20, 20:-20]
        # n, c, h, w = dcn_feat.shape
        # dcn_feat = torch.abs(dcn_feat)
        # dcn_max = dcn_feat.view(n, c, h * w).max(2, keepdim=True)[0].reshape(n, c, 1, 1)
        # dcn_feat = dcn_feat / (dcn_max.repeat(1, 1, h, w) + 1e-8)
        # a_dcn = torch.mean(dcn_feat, dim=1, keepdim=True)
        # a_dcn = torch.clamp(a_dcn, min=0.0, max=1.0)
        # #
        # dyn_feat = dyn_feat[:, :, 20:-20, 20:-20]
        # n, c, h, w = dyn_feat.shape
        # dyn_feat = torch.abs(dyn_feat)
        # dyn_max = dyn_feat.view(n, c, h * w).max(2, keepdim=True)[0].reshape(n, c, 1, 1)
        # dyn_feat = dyn_feat / (dyn_max.repeat(1, 1, h, w) + 1e-8)
        # a_dyn = torch.mean(dyn_feat, dim=1, keepdim=True)
        # a_dyn = torch.clamp(a_dyn, min=0.0, max=1.0)
        # return fused_feat, -1, [a_dcn, a_dyn]

        return fused_feat, -1, [0, 0]


# ==========
# Quality enhancement module
# ==========

class PlainCNN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()

        self.nb = nb
        self.in_nc = in_nc

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv_1{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                )
            )
            # setattr(
            #     self, 'dn_conv_dwt{}'.format(i), nn.Sequential(
            #         nn.Conv2d(3 * 3, nf, base_ks, padding=base_ks // 2),
            #         nn.ReLU(inplace=True),
            #         ResidualBlock_noBN(nf),
            #         )
            #     )
            # setattr(
            #     self, 'dn_conv_concat{}'.format(i), nn.Sequential(
            #         nn.Conv2d(nf * 2, nf, base_ks, padding=base_ks // 2),
            #         nn.ReLU(inplace=True),
            #         )
            #     )
            setattr(
                self, 'dn_conv_2{}'.format(i), nn.Sequential(
                    ResidualBlock_noBN(nf),
                    ResidualBlock_noBN(nf),
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    ResidualBlock_noBN(nf),
                    ResidualBlock_noBN(nf),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            ResidualBlock_noBN(nf),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, out_nc, base_ks, padding=base_ks // 2),
        )

    def forward(self, inputs, inputs_l, inputs_h):
        nb = self.nb
        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv_1 = getattr(self, 'dn_conv_1{}'.format(i))
            # dn_conv_dwt = getattr(self, 'dn_conv_dwt{}'.format(i))
            # dn_conv_concat = getattr(self, 'dn_conv_concat{}'.format(i))
            dn_conv_2 = getattr(self, 'dn_conv_2{}'.format(i))

            fea = dn_conv_1(out_lst[i - 1])
            # dwt_fea = dn_conv_dwt(inputs_h[i - 1])
            # concat_fea = dn_conv_concat(torch.cat([fea, dwt_fea], dim=1))
            out_lst.append(dn_conv_2(fea))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )
        out = self.out_conv(out)
        return out


# ==========
# MFVQE network
# ==========

class dpdd_s12_dunet_ab10(nn.Module):
    """STDF -> QE -> residual.

    in: (B T C H W)
    out: (B C H W)
    """

    def __init__(self, opts_dict):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(dpdd_s12_dunet_ab10, self).__init__()

        if 'use_align_loss' in opts_dict.keys():
            self.use_align_loss = opts_dict['use_align_loss']
        else:
            self.use_align_loss = True

        self.ffnet = STDF(
            in_nc=3,
            out_nc=opts_dict['stdf']['out_nc'],
            nf=opts_dict['stdf']['nf'],
            nb=opts_dict['stdf']['nb'],
            deform_ks=opts_dict['stdf']['deform_ks']
        )
        self.qenet = PlainCNN(
            in_nc=opts_dict['qenet']['in_nc'],
            out_nc=opts_dict['qenet']['out_nc'],
            nf=opts_dict['qenet']['nf'],
            nb=opts_dict['qenet']['nb'],
        )

        self.dwt = pytorch_wavelets.DWTForward(J=4, wave='haar')
        self.idwt = pytorch_wavelets.DWTInverse(wave='haar')

    def forward(self, x):
        xl, xh = self.dwt(x)
        # print(xl.shape, xh[0].shape, xh[1].shape, xh[2].shape)

        xls = xl
        xhs_0 = xh[0].flatten(1, 2)
        xhs_1 = xh[1].flatten(1, 2)
        xhs_2 = xh[2].flatten(1, 2)
        xhs_3 = xh[3].flatten(1, 2)

        xhs = [xhs_0, xhs_1, xhs_2, xhs_3]

        if self.training:
            out, recon, a = self.ffnet(x, xls, xhs)
            out = self.qenet(out, xls, xhs)
            out += x  # res: add middle frame
            # if self.use_align_loss:
            #     return out, recon
            # else:
            #     return out, -1
        else:
            out, recon, extra = self.ffnet(x, xls, xhs)
            out = self.qenet(out, xls, xhs)
            out += x  # res: add middle frame
            # return out, extra

        return out
