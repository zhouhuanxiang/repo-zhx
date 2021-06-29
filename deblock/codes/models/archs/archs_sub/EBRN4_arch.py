''' network architecture for EBRN '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

'''Compared to EBRN3
1. SR flow before Back-projection flow changed to channelwise convolution
2. init
3. 256 -> 64 
4. add long skip connection

'''
class Block_Residual_Module(nn.Module):
    def __init__(self, with_bp_flow=True):
        super(Block_Residual_Module, self).__init__()
        self.with_bp_flow = with_bp_flow

        self.sr_flow_before_split_1 = arch_util.ResidualBlock_noBN_channelwise(64)
        self.sr_flow_before_split_2 = arch_util.ResidualBlock_noBN_channelwise(64)
        self.sr_flow_after_split_1 = arch_util.ResidualBlock_noBN(64)
        # self.sr_flow_after_split_2 = arch_util.ResidualBlock_noBN(64)

        if self.with_bp_flow:
            # self.bp_flow_before_compare_1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
            self.bp_flow_after_compare_1 = arch_util.ResidualBlock_noBN(64)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, Ix):
        Ox_before_split = self.lrelu(self.sr_flow_before_split_1(Ix))
        Ox_before_split = self.lrelu(self.sr_flow_before_split_2(Ox_before_split))
        Ox_after_split = self.lrelu(self.sr_flow_after_split_1(Ox_before_split))
        # Ox_after_split = self.lrelu(self.sr_flow_after_split_2(Ox_after_split))

        if self.with_bp_flow:
            I_x1_after_compare = self.bp_flow_after_compare_1(Ox_before_split - Ix)
            return Ox_after_split, I_x1_after_compare
        else:
            return Ox_after_split


class EBRN(nn.Module):
    def __init__(self):
        super(EBRN, self).__init__()
        self.feature_extration_1 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        # self.feature_extration_2 = nn.Conv2d(256, 64, 3, 1, 1, bias=True)
        # self.feature_extration_3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.feature_extration_2 = arch_util.ResidualBlock_noBN(64)

        self.brm1 = Block_Residual_Module(True)
        self.brm2 = Block_Residual_Module(True)
        self.brm3 = Block_Residual_Module(True)
        self.brm4 = Block_Residual_Module(True)
        self.brm5 = Block_Residual_Module(False)

        self.fusion45 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.fusion34 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.fusion23 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.fusion12 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)

        self.recon1 = nn.Conv2d(64 * 5, 64, 3, 1, 1, bias=True)
        self.recon2 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.feature_extration_1,
                            self.fusion45, self.fusion34, self.fusion23, self.fusion12,
                            self.recon1, self.recon2],
                           0.1)

    def forward(self, x):
        I = self.lrelu(self.feature_extration_1(x))
        I = self.lrelu(self.feature_extration_2(I))

        O1, I = self.brm1(I)
        O2, I = self.brm2(I)
        O3, I = self.brm3(I)
        O4, I = self.brm4(I)
        O5 = self.brm5(I)

        O4 = self.lrelu(self.fusion45(O4 + O5))
        O3 = self.lrelu(self.fusion34(O3 + O4))
        O2 = self.lrelu(self.fusion23(O2 + O3))
        O1 = self.lrelu(self.fusion12(O1 + O2))

        O_concat = torch.cat((O1, O2, O3, O4, O5), dim=1)
        y = self.lrelu(self.recon1(O_concat))
        y = self.recon2(y) + x

        return y



