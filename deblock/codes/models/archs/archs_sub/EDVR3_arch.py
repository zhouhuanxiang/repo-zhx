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


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8, w_GCB=False, return_offset=False):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True, return_offset=return_offset)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True, return_offset=return_offset)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True, return_offset=return_offset)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True, return_offset=return_offset)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.w_GCB = w_GCB
        self.return_offset = return_offset
        if self.w_GCB:
            self.L1_context_block = context_block.ContextBlock(64, 0.25)
            self.L2_context_block = context_block.ContextBlock(64, 0.25)
            self.L3_context_block = context_block.ContextBlock(64, 0.25)


    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        if self.return_offset:
            L3_fea, L3_offset_return = self.L3_dcnpack([nbr_fea_l[2], L3_offset])
        else:
            L3_fea = self.L3_dcnpack([nbr_fea_l[2], L3_offset])
        L3_fea = self.lrelu(L3_fea)
        if self.w_GCB:
            L3_fea = self.L3_context_block(L3_fea)
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        if self.return_offset:
            L2_fea, L2_offset_return = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        else:
            L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L2_fea = self.lrelu(L2_fea)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        if self.w_GCB:
            L2_fea = self.L2_context_block(L2_fea)
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        if self.return_offset:
            L1_fea, L1_offset_return = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        else:
            L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L1_fea = self.lrelu(L1_fea)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        if self.w_GCB:
            L1_fea = self.L1_context_block(L1_fea)
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        if self.return_offset:
            L1_fea, L0_offset_return = self.cas_dcnpack([L1_fea, offset])
        else:
            L1_fea = self.cas_dcnpack([L1_fea, offset])
        L1_fea = self.lrelu(L1_fea)

        if self.return_offset:
            # return L1_fea, [L0_offset_return, L1_offset_return, L2_offset_return, L3_offset_return]
            return L1_fea, L0_offset_return
        else:
            return L1_fea


class EDVR(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True, w_GCB=False, offset_only=False):
        super(EDVR, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        self.offset_only = offset_only
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        else:
            if self.HR_in:
                self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            else:
                self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PCD_Align(nf=nf, groups=groups, w_GCB=w_GCB, return_offset=offset_only)
        # if self.w_TSA:
        #     self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        # else:
        #     self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        # self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        base = x

        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(x.view(-1, C, H, W))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H, W = H // 4, W // 4
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H, W = H // 4, W // 4
            else:
                L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### pcd align
        # ref feature list
        if not self.offset_only:
            ref_fea_l = [
                L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
                L3_fea[:, self.center, :, :, :].clone()
            ]
            out_l = []
            for i in range(N):
                nbr_fea_l = [
                    L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                    L3_fea[:, i, :, :, :].clone()
                ]
                out = self.pcd_align(nbr_fea_l, ref_fea_l)
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.lrelu(self.HRconv(out))
                out = self.conv_last(out)
                out_l.append(out)
            out_l = torch.stack(out_l, dim=1)  # [B, N, C, H, W]

            out_l += base
            # return out, self.aligned_fea_detach
            return out_l
        else:
            ref_fea_l = [
                L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
                L3_fea[:, self.center, :, :, :].clone()
            ]
            out_l = []
            for i in range(N):
                nbr_fea_l = [
                    L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                    L3_fea[:, i, :, :, :].clone()
                ]
                _, out = self.pcd_align(nbr_fea_l, ref_fea_l)
                out_l.append(out)
            out_l = torch.stack(out_l, dim=1)

            return out_l
