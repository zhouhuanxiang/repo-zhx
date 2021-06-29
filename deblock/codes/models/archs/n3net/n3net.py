'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import math
import functools
import torch
import torch.nn as nn
import models.archs.arch_util as arch_util

from . import non_local

def dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025):
    r"""
    Reproduces batchnorm initialization from DnCNN
    https://github.com/cszn/DnCNN/blob/master/TrainingCodes/DnCNN_TrainingCodes_v1.1/DnCNN_init_model_64_25_Res_Bnorm_Adam.m
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.weight.data[(m.weight.data > 0) & (m.weight.data <= b_min)] = b_min
    m.weight.data[(m.weight.data < 0) & (m.weight.data >= -b_min)] = -b_min
    m.weight.data = m.weight.data.abs()
    m.bias.data.zero_()
    m.momentum = 0.001

def cnn_from_def(cnn_opt):
    kernel = cnn_opt.get("kernel",3)
    padding = (kernel-1)//2
    cnn_bn = cnn_opt.get("bn",True)
    cnn_depth = cnn_opt.get("depth",0)
    cnn_channels = cnn_opt.get("features")
    cnn_outchannels = cnn_opt.get("nplanes_out",)
    chan_in = cnn_opt.get("nplanes_in")

    if cnn_depth == 0:
        cnn_outchannels=chan_in

    cnn_layers = []
    relu = nn.ReLU(inplace=True)

    for i in range(cnn_depth-1):
        cnn_layers.extend([
            nn.Conv2d(chan_in,cnn_channels,kernel, 1, padding, bias=not cnn_bn),
            nn.BatchNorm2d(cnn_channels) if cnn_bn else nn.Sequential(),
            relu
        ])
        chan_in = cnn_channels

    if cnn_depth > 0:
        cnn_layers.append(
            nn.Conv2d(chan_in,cnn_outchannels,kernel, 1, padding, bias=True)
        )

    net = nn.Sequential(*cnn_layers)
    net.nplanes_out = cnn_outchannels
    net.nplanes_in = cnn_opt.get("nplanes_in")
    return net


class N3Block(nn.Module):
    r"""
    N3Block operating on a 2D images
    """
    def __init__(self, nplanes_in, k, patchsize=10, stride=5,
                 nl_match_window=15,
                 temp_opt={}, embedcnn_opt={}):
        r"""
        :param nplanes_in: number of input features
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param nl_match_window: size of matching window around each patch,
            i.e. the nl_match_window x nl_match_window patches around a query patch
            are used for matching
        :param temp_opt: options for handling the the temperature parameter
        :param embedcnn_opt: options for the embedding cnn, also shared by temperature cnn
        """
        super(N3Block, self).__init__()
        self.patchsize = patchsize
        self.stride = stride

        # patch embedding
        embedcnn_opt["nplanes_in"] = nplanes_in
        self.embednet = cnn_from_def(embedcnn_opt)

        # temperature cnn
        with_temp = temp_opt.get("external_temp")
        if with_temp:
            tempcnn_opt = dict(**embedcnn_opt)
            tempcnn_opt["nplanes_out"] = 1
            self.tempcnn = cnn_from_def(tempcnn_opt)
        else:
            self.tempcnn = None

        self.nplanes_in = nplanes_in
        self.nplanes_out = (k+1) * nplanes_in

        indexer = lambda xe_patch,ye_patch: non_local.index_neighbours(xe_patch, ye_patch, nl_match_window, exclude_self=True)
        self.n3aggregation = non_local.N3Aggregation2D(indexing=indexer, k=k,
                patchsize=patchsize, stride=stride, temp_opt=temp_opt)
        self.k = k

        self.reset_parameters()

    def forward(self, x, y):
        if self.k <= 0:
            return x

        xe = self.embednet(x)
        ye = self.embednet(y)

        xg = x
        if self.tempcnn is not None:
            log_temp = self.tempcnn(x)
        else:
            log_temp = None

        x = self.n3aggregation(xg,xe,ye,y,log_temp=log_temp)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d)):
                dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025)

class N3NetVideo(nn.Module):
    r"""
    A N3Net modified for video
    """
    def __init__(self, nf, nframes, front_RBs, back_RBs):
        super(N3NetVideo, self).__init__()
        self.center = nframes // 2

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.feature_extraction = arch_util.make_layer(basic_block, front_RBs)

        k = 7
        nl_opt = dict(
            k=7,
            patchsize=16,
            stride=8,
            temp_opt=dict(
                external_temp=True,
                temp_bias=0.1,
                distance_bn=True,
                avgpool=True
            ),
            embedcnn_opt=dict(
                kernel=3,
                bn=True,
                depth=3,
                features=8,
                nplanes_out=8,
                nplanes_in=nf
            )
        )
        self.nl = N3Block(nf, **nl_opt)
        self.patch_fusion = nn.Conv2d(nf * (k + 1), nf, 3, 1, 1, bias=True)
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.recon_trunk = arch_util.make_layer(basic_block, back_RBs)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        L1_fea = L1_fea.view(B, N, -1, H, W).contiguous()

        aligned_fea = []
        for i in range(N):
            aligned_fea_tmp = self.nl(L1_fea[:, i, :, :, :].contiguous(), L1_fea[:, self.center, :, :, :].contiguous())
            aligned_fea_tmp = self.patch_fusion(aligned_fea_tmp)
            aligned_fea.append(aligned_fea_tmp)
        aligned_fea = torch.stack(aligned_fea, dim=1)
        aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.recon_trunk(fea)
        out = self.conv_last(out)
        out += x_center
        return out

class N3Net(nn.Module):
    r"""
    A N3Net interleaves DnCNNS for local processing with N3Blocks for non-local processing
    """
    def __init__(self, nplanes_in, nplanes_out, nplanes_interm, nblocks, block_opt, nl_opt, residual=False):
        r"""
        :param nplanes_in: number of input features
        :param nplanes_out: number of output features
        :param nplanes_interm: number of intermediate features, i.e. number of output features for the DnCNN sub-networks
        :param nblocks: number of DnCNN sub-networks
        :param block_opt: options passed to DnCNNs
        :param nl_opt: options passed to N3Blocks
        :param residual: whether to have a global skip connection
        """
        super(N3Net, self).__init__()
        self.nplanes_in = nplanes_in
        self.nplanes_out = nplanes_out
        self.nblocks = nblocks
        self.residual = residual

        nin = nplanes_in
        cnns = []
        nls = []
        for i in range(nblocks-1):
            cnns.append(DnCNN(nin, nplanes_interm, **block_opt))
            nl = N3Block(nplanes_interm, **nl_opt)
            nin = nl.nplanes_out
            nls.append(nl)

        nout = nplanes_out
        cnns.append(DnCNN(nin, nout, **block_opt))

        self.nls = nn.Sequential(*nls)
        self.blocks = nn.Sequential(*cnns)

    def forward(self, x):
        shortcut = x
        for i in range(self.nblocks-1):
            x = self.blocks[i](x)
            x = self.nls[i](x)

        x = self.blocks[-1](x)

        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out)
            x[:,:nshortcut,:,:] = x[:,:nshortcut,:,:] + shortcut[:,:nshortcut,:,:]

        return x
