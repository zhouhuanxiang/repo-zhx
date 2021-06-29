''' network architecture for EDVR '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import models.archs.context_block as context_block
try:
    from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

import pytorch_wavelets

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


class Predeblur_ResNet_Pyramid(nn.Module):
    def __init__(self, nf=128, HR_in=False):
        '''
        HR_in: True if the inputs are high spatial size
        '''

        super(Predeblur_ResNet_Pyramid, self).__init__()
        self.HR_in = True if HR_in else False
        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.RB_L1_1 = basic_block()
        self.RB_L1_2 = basic_block()
        self.RB_L1_3 = basic_block()
        self.RB_L1_4 = basic_block()
        self.RB_L1_5 = basic_block()
        self.RB_L2_1 = basic_block()
        self.RB_L2_2 = basic_block()
        self.RB_L3_1 = basic_block()
        self.deblur_L2_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.deblur_L3_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        if self.HR_in:
            L1_fea = self.lrelu(self.conv_first_1(x))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = F.interpolate(self.RB_L3_1(L3_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = F.interpolate(self.RB_L2_2(L2_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8, w_GCB=False):
        super(PCD_Align, self).__init__()
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf * 2, nf * 2, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf * 2, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, nbr_fea_l, nbr_fea_l_2, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0], nbr_fea_l_2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        a = torch.cat([nbr_fea_l[0], nbr_fea_l_2[0]], dim=1)
        print(a.shape, L1_offset.shape)
        L1_fea = self.L1_dcnpack([torch.cat([nbr_fea_l[0], nbr_fea_l_2[0]], dim=1), L1_offset])
        L1_fea = self.L1_fea_conv(L1_fea)
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea


class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea


class EDVR(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True, w_GCB=False):
        super(EDVR, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        self.conv_first = nn.Conv2d(3 * 16, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)

        self.pcd_align = PCD_Align(nf=nf, groups=groups, w_GCB=w_GCB)
        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion = nn.Conv2d((nframes // 2 + 1) * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3 * 16, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        ####
        self.dwt_foward = DWTForward()
        self.dwt_inverse = DWTInverse()

    def forward(self, x):

        x_center = x[:, self.center, :, :, :].contiguous()
        x = self.dwt_foward(self.dwt_foward(x))
        B, N, C, H, W = x.size()  # N video frames

        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)

        L1_fea = L1_fea.view(B, N, -1, H, W)

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N // 2 + 1):
            nbr_fea_l_1 = [
                L1_fea[:, i, :, :, :].clone()
            ]
            nbr_fea_l_2 = [
                L1_fea[:, N - 1 - i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l_1, nbr_fea_l_2, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        # self.aligned_fea_detach = aligned_fea.detach()
        # self.aligned_fea_detach = torch.mean(self.aligned_fea_detach, 2, keepdim=True)
        # self.aligned_fea_detach = torch.max(torch.abs(self.aligned_fea_detach, 2), 2, keepdim=True)[0]
        # self.aligned_fea_detach = self.aligned_fea_detach[:,:,0,:,:]

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.recon_trunk(fea)
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)

        out = self.dwt_inverse(self.dwt_inverse(out))

        out += x_center
        # return out, self.aligned_fea_detach
        return out
