import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import functools
import models.archs.arch_util as arch_util
from models.archs.FAC.kernelconv2d import KernelConv2D

# class KernelConv(nn.Module):
#     """
#     the class of computing prediction
#     """
#     def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
#         super(KernelConv, self).__init__()
#         self.kernel_size = [5]
#         self.sep_conv = sep_conv
#         self.core_bias = core_bias
#
#     def _convert_dict(self, core, batch_size, N, color, height, width):
#         """
#         make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
#         :param core: shape: batch_size*(N*K*K)*height*width
#         :return: core_out, a dict
#         """
#         core_out = {}
#         core = core.view(batch_size, N, -1, color, height, width)
#         core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
#         bias = None if not self.core_bias else core[:, :, -1, ...]
#         return core_out, bias
#
#     def forward(self, frames, core):
#         """
#         compute the pred image according to core and frames
#         :param frames: [batch_size, N, 3, height, width]
#         :param core: [batch_size, N, dict(kernel), 3, height, width]
#         :return:
#         """
#         if len(frames.size()) == 5:
#             batch_size, N, color, height, width = frames.size()
#         else:
#             batch_size, N, height, width = frames.size()
#             color = 1
#             frames = frames.view(batch_size, N, color, height, width)
#         if self.sep_conv:
#             core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
#         else:
#             core, bias = self._convert_dict(core, batch_size, N, color, height, width)
#         img_stack = []
#         pred_img = []
#         kernel = self.kernel_size[::-1]
#         for index, K in enumerate(kernel):
#             if not img_stack:
#                 frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
#                 for i in range(K):
#                     for j in range(K):
#                         img_stack.append(frame_pad[..., i:i + height, j:j + width])
#                 img_stack = torch.stack(img_stack, dim=2)
#             else:
#                 k_diff = (kernel[index - 1] - kernel[index]) // 2
#                 img_stack = img_stack[:, :, k_diff:-k_diff, ...]
#             # print('img_stack:', img_stack.size())
#             pred_img.append(torch.sum(
#                 core[K].mul(img_stack), dim=2, keepdim=False
#             ))
#         pred_img = torch.stack(pred_img, dim=0)
#         pred_img_i = torch.mean(pred_img, dim=0, keepdim=False).squeeze()
#         return pred_img_i

# if __name__ == '__main__':
#     kernel_conv = KernelConv()
#     frames = torch.rand(4, 1, 3, 224, 224)
#     core = torch.rand(4, 1, 5*5, 3, 224, 224)
#     re = kernel_conv(frames, core)
#     print(re.shape)

class DynamicBlock(nn.Module):
    def __init__(self):
        super(DynamicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 3*5*5, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        # self.kernel_conv = KernelConv()
        self.kconv_conv = KernelConv2D.KernelConv2D(kernel_size=5)

    def forward(self, I, fea):
        fea = torch.cat([self.conv3(I), fea], dim=1)
        residual = self.conv2(fea)
        core = self.conv1(fea)

        # B, C, H, W = I.shape
        # I = I.reshape(B, 1, C, H, W)
        # core = core.reshape(B, 1, 5*5, C, H, W)
        # out = self.kernel_conv(I, core)

        out = self.kconv_conv(I, core)

        out += residual

        return out


class MSRResNetKPN(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNetKPN, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first],
                                     0.1)

        self.dynamic_block_1 = DynamicBlock()


    def forward(self, x):
        # Feature Extraction Network
        fea1 = self.lrelu(self.conv_first(x))
        fea = self.recon_trunk(fea1)
        fea += fea1
        # Dynamic Block
        x = self.dynamic_block_1(x, fea)

        return x
