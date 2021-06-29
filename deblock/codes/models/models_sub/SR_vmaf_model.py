import logging
from collections import OrderedDict

import os
import re
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, ContextualLoss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import utils.util as util
from data.util import run_vmaf_pytorch, run_vmaf_pytorch_parallel
from xml.dom import minidom
import multiprocessing
import time

logger = logging.getLogger('base')


class SRVmafModel(BaseModel):
    def __init__(self, opt):
        super(SRVmafModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.use_gpu = opt['network_G']['use_gpu']
        self.use_gpu = True
        self.real_IQA_only = train_opt['IQA_only']

        # define network and load pretrained models
        if self.use_gpu:
            self.netG = networks.define_G(opt).to(self.device)
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            else:
                self.netG = DataParallel(self.netG)
        else:
            self.netG = networks.define_G(opt)

        if self.is_train:
            if train_opt['IQA_weight']:
                if train_opt['IQA_criterion'] == 'vmaf':
                    self.cri_IQA = nn.MSELoss()
                self.l_IQA_w = train_opt['IQA_weight']

                self.netI = networks.define_I(opt)
                if opt['dist']:
                    pass
                else:
                    self.netI = DataParallel(self.netI)
            else:
                logger.info('Remove IQA loss.')
                self.cri_IQA = None

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # pixel loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # CX loss
            if train_opt['CX_weight']:
                l_CX_type = train_opt['CX_criterion']
                if l_CX_type == 'contextual_loss':
                    self.cri_CX = ContextualLoss()
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_CX_type))
                self.l_CX_w = train_opt['CX_weight']
            else:
                logger.info('Remove CX loss.')
                self.cri_CX = None

            # ssim loss
            if train_opt['ssim_weight']:
                self.cri_ssim = train_opt['ssim_criterion']
                self.l_ssim_w = train_opt['ssim_weight']
                self.ssim_window = train_opt['ssim_window']
            else:
                logger.info('Remove ssim loss.')
                self.cri_ssim = None


            # load VGG perceptual loss if use CX loss
            # if train_opt['CX_weight']:
            #     self.netF = networks.define_F(opt, use_bn=False).to(self.device)
            #     if opt['dist']:
            #         pass  # do not need to use DistributedDataParallel for netF
            #     else:
            #         self.netF = DataParallel(self.netF)

            # optimizers of netG
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # optimizers of netI
            if train_opt['IQA_weight']:
                wd_I = train_opt['weight_decay_I'] if train_opt['weight_decay_I'] else 0
                optim_params = []
                for k, v in self.netI.named_parameters():  # can optimize for a part of the model
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                self.optimizer_I = torch.optim.Adam(optim_params, lr=train_opt['lr_I'],
                                                    weight_decay=wd_I,
                                                    betas=(train_opt['beta1'], train_opt['beta2']))
                self.optimizers.append(self.optimizer_I)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
            self.set_requires_grad(self.netG, False)
            self.set_requires_grad(self.netI, False)

    def feed_data(self, data, need_GT=True):
        if self.use_gpu:
            self.var_L = data['LQ'].to(self.device)  # LQ
            if need_GT:
                self.real_H = data['GT'].to(self.device)  # GT
            if self.cri_IQA and ('IQA' in data.keys()):
                self.real_IQA = data['IQA'].float().to(self.device) # IQA
        else:
            self.var_L = data['LQ']  # LQ

    def optimize_parameters(self, step):
        #init loss
        l_pix = torch.zeros(1)
        l_CX = torch.zeros(1)
        l_ssim = torch.zeros(1)
        l_g_IQA = torch.zeros(1)
        l_i_IQA = torch.zeros(1)

        if self.cri_IQA and self.real_IQA_only:
            # pretrain netI
            self.set_requires_grad(self.netI, True)
            self.optimizer_I.zero_grad()

            iqa = self.netI(self.var_L, self.real_H).squeeze()
            l_i_IQA = self.l_IQA_w * self.cri_IQA(iqa, self.real_IQA)
            l_i_IQA.backward()

            self.optimizer_I.step()
        elif self.cri_IQA and not self.real_IQA_only:
            # train netG and netI together

            # optimize netG
            self.set_requires_grad(self.netG, True)
            self.optimizer_G.zero_grad()

            # forward
            self.fake_H = self.netG(self.var_L)

            l_g_total = 0
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            l_g_total += l_pix
            if self.cri_CX:
                real_fea = self.netF(self.real_H)
                fake_fea = self.netF(self.fake_H)
                l_CX = self.l_CX_w * self.cri_CX(real_fea, fake_fea)
                l_g_total += l_CX
            if self.cri_ssim:
                if self.cri_ssim == 'ssim':
                    ssim_val = ssim(self.fake_H, self.real_H, win_size=self.ssim_window, data_range=1.0, size_average=True)
                elif self.cri_ssim == 'ms-ssim':
                    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363]).to(self.fake_H.device, dtype=self.fake_H.dtype)
                    ssim_val = ms_ssim(self.fake_H, self.real_H, win_size=self.ssim_window, data_range=1.0, size_average=True, weights=weights)
                l_ssim = self.l_ssim_w * (1 - ssim_val)
                l_g_total += l_ssim
            if self.cri_IQA:
                l_g_IQA = self.l_IQA_w * (1.0 - torch.mean(self.netI(self.fake_H, self.real_H)))
                l_g_total += l_g_IQA

            l_g_total.backward()
            self.optimizer_G.step()
            self.set_requires_grad(self.netG, False)

            # optimize netI
            self.set_requires_grad(self.netI, True)
            self.optimizer_I.zero_grad()

            self.fake_H_detatch = self.fake_H.detach()

            # t1 = time.time()
            # real_IQA1 = run_vmaf_pytorch(self.fake_H_detatch, self.real_H)
            # t2 = time.time()
            real_IQA2 = run_vmaf_pytorch_parallel(self.fake_H_detatch, self.real_H)
            # t3 = time.time()
            # print(real_IQA1)
            # print(real_IQA2)
            # print(t2 - t1, t3 - t2, '\n')
            real_IQA = real_IQA2.to(self.device)

            iqa = self.netI(self.fake_H_detatch, self.real_H).squeeze()
            l_i_IQA = self.cri_IQA(iqa, real_IQA)
            l_i_IQA.backward()

            self.optimizer_I.step()
            self.set_requires_grad(self.netI, False)


        # set log
        self.log_dict['l_pix'] = l_pix.item()
        if self.cri_CX:
            self.log_dict['l_CX'] = l_CX.item()
        if self.cri_ssim:
            self.log_dict['l_ssim'] = l_ssim.item()
        if self.cri_IQA:
            self.log_dict['l_g_IQA_scale'] = l_g_IQA.item()
            self.log_dict['l_g_IQA'] = l_g_IQA.item() / self.l_IQA_w
            self.log_dict['l_i_IQA'] = l_i_IQA.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        if self.use_gpu:
            out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
            out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        else:
            out_dict['LQ'] = self.var_L.detach()[0].float()
            out_dict['rlt'] = self.fake_H.detach()[0].float()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

        load_path_I = self.opt['path']['pretrain_model_I']
        if load_path_I is not None:
            logger.info('Loading model for I [{:s}] ...'.format(load_path_I))
            self.load_network(load_path_I, self.netI, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netI, 'I', iter_label)