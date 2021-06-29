import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss

logger = logging.getLogger('base')


class VideoAttentionModel(BaseModel):
    def __init__(self, opt):
        super(VideoAttentionModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            train_opt = opt['train']
            self.patch_size = opt['datasets']['train']['patch_size']
            self.patch_repeat = opt['datasets']['train']['patch_repeat']
            self.use_diff = opt['datasets']['train']['use_diff']

            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'bce':
                self.cri_pix = nn.BCEWithLogitsLoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
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
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        if self.is_train:
            self.var_L = data['LQs'].to(self.device)
            self.diff_L = data['Diffs'].to(self.device)
            self.patch_labels = data['patch_labels']
            self.patch_offsets = data['patch_offsets']
        else:
            self.var_L = data['LQs'].to(self.device)

        # if need_GT:
        #     self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.var_L = self.var_L.view(self.var_L.shape[0], -1, self.var_L.shape[3], self.var_L.shape[4])
        self.fake_H = self.netG(self.var_L)

        self.fake_H = torch.squeeze(self.fake_H, 1)
        mask_H = torch.zeros(self.fake_H.shape)
        label_H = torch.zeros(self.fake_H.shape)
        for i in range(self.fake_H.shape[0]):
            for j in range(self.patch_repeat):
                offset_h = self.patch_offsets[i][j][0].item()
                offset_w = self.patch_offsets[i][j][1].item()
                label = self.patch_labels[i][j].item()
                mask_H[i, offset_h:offset_h + self.patch_size, offset_w:offset_w + self.patch_size] \
                    = torch.ones(self.patch_size, self.patch_size)
                label_H[i, offset_h:offset_h + self.patch_size, offset_w:offset_w + self.patch_size] \
                    = label * torch.ones(self.patch_size, self.patch_size)

        mask_H = mask_H.to(self.device)
        label_H = label_H.to(self.device)

        # l_pix = self.l_pix_w * torch.sum(self.cri_pix(torch.mul(self.fake_H, mask_H), torch.mul(label_H, mask_H))) / torch.sum(mask_H)

        self.fake_H = torch.mul(self.fake_H, mask_H)
        label_H = torch.mul(label_H, mask_H)
        if self.use_diff:
            self.diff_L = torch.mean(torch.abs(torch.squeeze(self.diff_L, 1)), 1)
            self.fake_H = torch.mul(self.fake_H, self.diff_L)
            label_H = torch.mul(label_H, self.diff_L)
        l_pix = torch.sum(self.cri_pix(self.fake_H, label_H)) / torch.sum(mask_H)
        print(torch.sum(mask_H))

        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            assert self.var_L.shape[1] == 7, 'frame window shound be 7'
            center_frame = self.var_L[:, 4, :, :, :]
            self.fake_H = []
            for v in range(7):
                if v == 3:
                    self.fake_H.append(center_frame)
                else:
                    neighbor_frame = self.var_L[:, v, :, :, :]
                    neighbor_frame = torch.cat((center_frame, neighbor_frame), 1)
                    fake_H = self.netG(neighbor_frame)
                    fake_H = fake_H.repeat(1, 3, 1, 1)
                    self.fake_H.append(fake_H)
            self.fake_H = torch.stack(self.fake_H, dim=1)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
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

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
