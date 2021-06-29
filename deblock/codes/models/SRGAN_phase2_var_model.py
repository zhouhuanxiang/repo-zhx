import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import GANLoss, GradientPenaltyLoss, tv_loss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

logger = logging.getLogger('base')


class SRGANPhase2VarModel(BaseModel):
    def __init__(self, opt):
        super(SRGANPhase2VarModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        import models.archs.SRResNet_arch as SRResNet_arch
        self.netG_phase1 = SRResNet_arch.MSRResNet(in_nc=3, out_nc=3, nf=64, nb=16, upscale=1)
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.netG_phase1 = DistributedDataParallel(self.netG_phase1, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG).to(self.device)
            self.netG_phase1 = DataParallel(self.netG_phase1).to(self.device)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD,
                                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD).to(self.device)

            self.netG.train()
            self.netG_phase1.eval()
            self.netD.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # ssim loss
            if train_opt['ssim_weight']:
                self.cri_ssim = train_opt['ssim_criterion']
                self.l_ssim_w = train_opt['ssim_weight']
                self.ssim_window = train_opt['ssim_window']
            else:
                logger.info('Remove ssim loss.')
                self.cri_ssim = None

            # perceptual loss
            if train_opt['loss_feature_weight'] > 0:
                loss_fea_type = train_opt['loss_feature_type']
                if loss_fea_type == 'l1':
                    self.loss_fea = nn.L1Loss().to(self.device)
                elif loss_fea_type == 'l2':
                    self.loss_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_fea_type))
                self.loss_fea_w = train_opt['loss_feature_weight']
            else:
                print('Remove feature loss.')
                self.loss_fea = None
            if self.loss_fea:  # load VGG
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # gan loss
            self.loss_gan = GANLoss(train_opt['loss_gan_type'], 1.0, 0.0).to(self.device)
            self.loss_gan_w = train_opt['loss_gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = 1
            self.D_init_iters = 0

            if train_opt['loss_gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.loss_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.loss_gp_w = train_opt['gp_weigth']

            # optimizers
            # G
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
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

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

        self.print_network()  # print network
        self.load()  # load G and D if needed
        self.w_phase1 = True
        if self.w_phase1:
            # self.load_network('/home/web_server/zhouhuanxiang/disk/log/experiments/train_EDSR_Xiph3/models/latest_G.pth',
            #                   self.netG_phase1, True)
            self.load_network('/home/web_server/zhouhuanxiang/disk/log/experiments/baseline/SRResNet_KWAI253035_baseline/models/latest_G.pth',
                              self.netG_phase1, True)
        if 'one2many' in self.opt['network_G'].keys():
            print('############ one2many #############')
            self.one2many = self.opt['network_G']['one2many']
        else:
            self.one2many = False

        if 'sa' in self.opt['network_G'].keys():
            print('############ sa #############')
            self.sa = self.opt['network_G']['sa']
        else:
            self.sa = False

        if 'var' in self.opt['network_G'].keys():
            print('############ var #############')
            self.use_var = self.opt['network_G']['var']
        else:
            self.use_var = False

        if train_opt['loss_tv_weight']:
            print('############ tv #############')
            self.loss_tv_weight = train_opt['loss_tv_weight']
        else:
            self.loss_tv_weight = 0

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if self.use_var:
            self.variance_map = data['LQ_sigma'].to(self.device)
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        if self.use_var:
            if self.w_phase1:
                self.fake_H = self.netG(self.netG_phase1(self.var_L), self.variance_map)
            else:
                self.fake_H = self.netG(self.var_L, self.variance_map)
        else:
            if self.w_phase1:
                self.fake_H = self.netG(self.netG_phase1(self.var_L))
            else:
                self.fake_H = self.netG(self.var_L)

        loss_total_g = 0
        self.Output = self.fake_H
        self.ground_truth = self.var_ref
        self.loss_dis_w = self.l_pix_w
        self.loss_dis = self.cri_pix
        if step % self.D_update_ratio == 0 and step >= self.D_init_iters:
            # Train G
            if self.loss_dis:  # distortion loss
                loss_dis_g = self.loss_dis_w * self.loss_dis(self.Output, self.ground_truth)
                loss_total_g = loss_dis_g
            if self.loss_fea:  # perceptual loss
                real_fea = self.netF(self.ground_truth).detach()
                fake_fea = self.netF(self.Output)
                loss_fea_g = self.loss_fea_w * self.loss_fea(fake_fea, real_fea)
                loss_total_g += loss_fea_g
            if self.cri_ssim:
                if self.cri_ssim == 'ssim':
                    ssim_val = ssim(self.fake_H, self.ground_truth, win_size=self.ssim_window, data_range=1.0,
                                    size_average=True)
                elif self.cri_ssim == 'ms-ssim':
                    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363]).to(self.fake_H.device,
                                                                                     dtype=self.fake_H.dtype)
                    ssim_val = ms_ssim(self.fake_H, self.ground_truth, win_size=self.ssim_window, data_range=1.0,
                                       size_average=True, weights=weights)
                l_ssim = self.l_ssim_w * (1 - ssim_val)
                loss_total_g += l_ssim
            if self.loss_tv_weight > 0:
                l_tv = self.loss_tv_weight * tv_loss(self.fake_H, self.variance_map)
                loss_total_g += l_tv

            # gan loss
            pred_g_fake = self.netD(self.Output)
            if self.opt['train']['loss_gan_type'] == 'ragan':
                pred_g_real = self.netD(self.ground_truth).detach()
                loss_gan_g = self.loss_gan_w * self.loss_gan(pred_g_fake - pred_g_real.mean(0, keepdim=True), True)
            else:
                loss_gan_g = self.loss_gan_w * self.loss_gan(pred_g_fake, True)  # wgan:-E_g(D(x)), to min wasserstein distance:L=E_r(D(x))-E_g(D(x))
            loss_total_g += loss_gan_g

            loss_total_g.backward()
            self.optimizer_G.step()

        # Train D
        self.netD.zero_grad()
        loss_total_d = 0
        pred_d_real = self.netD(self.ground_truth)
        pred_d_fake = self.netD(self.Output.detach())
        if self.opt['train']['loss_gan_type'] == 'ragan':
            loss_gan_d_real = self.loss_gan(pred_d_real - pred_d_fake.mean(0, keepdim=True), True)  # wgan:-E_r(D(x))
            loss_gan_d_fake = self.loss_gan(pred_d_fake - pred_d_real.mean(0, keepdim=True), False)  # wgan:E_g(D(x))
        else:
            loss_gan_d_real = self.loss_gan(pred_d_real, True)  # wgan:-E_r(D(x))
            loss_gan_d_fake = self.loss_gan(pred_d_fake, False)  # wgan:E_g(D(x))

        loss_total_d = loss_gan_d_fake + loss_gan_d_real

        if self.opt['train']['loss_gan_type'] == 'wgan-gp':
            alpha = torch.randn(self.Output.size(0), 1, 1, 1).to(self.device)
            interp = (alpha * self.ground_truth + ((1 - alpha) * self.Output.detach())).requires_grad_(True)
            interp_crit = self.netD(interp)

            loss_gp_d = self.loss_gp_w * self.loss_gp(interp, interp_crit)
            loss_total_d += loss_gp_d

        loss_total_d.backward(retain_graph=True)
        self.optimizer_D.step()

        # set log
        # D
        self.log_dict['l_gan_d_real'] = loss_gan_d_real.item()
        self.log_dict['l_gan_d_fake'] = loss_gan_d_fake.item()
        self.log_dict['l_total_d'] = loss_total_d.item()

        if self.opt['train']['loss_gan_type'] == 'wgan-gp':
            self.log_dict['l_gp_d'] = loss_gp_d.item()
        # D outputs
        # self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        # self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())
        if step % self.D_update_ratio == 0 and step >= self.D_init_iters:
            # G
            if self.loss_dis:
                self.log_dict['loss_dis_g'] = loss_dis_g.item()
            if self.loss_fea:
                self.log_dict['loss_fea_g'] = loss_fea_g.item()
            if self.cri_ssim:
                self.log_dict['l_ssim'] = l_ssim.item()
            if self.loss_tv_weight > 0:
                self.log_dict['l_tv'] = l_tv.item()
            self.log_dict['loss_gan_g'] = loss_gan_g.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.sa:
                print(self.var_L.shape)
                z = torch.randn(self.var_L.shape[0], 1, self.var_L.shape[2] // 2, self.var_L.shape[3]).to(self.device)
                if self.w_phase1:
                    self.fake_H_1 = self.netG(self.netG_phase1(self.var_L[:, :, :self.var_L.shape[2] // 2, :]))
                    self.fake_H_2 = self.netG(self.netG_phase1(self.var_L[:, :, self.var_L.shape[2] // 2:, :]))
                else:
                    self.fake_H_1 = self.netG(torch.cat(self.var_L[:, :, :self.var_L.shape[2] // 2:, :]))
                    self.fake_H_2 = self.netG(torch.cat(self.var_L[:, :, self.var_L.shape[2] // 2:, :]))
                self.fake_H = torch.cat((self.fake_H_1, self.fake_H_2), dim=2)
                print(self.fake_H_1.shape, self.fake_H_2.shape, self.fake_H.shape)
            else:
                if self.use_var:
                    if self.w_phase1:
                        self.fake_H = self.netG(self.netG_phase1(self.var_L), self.variance_map)
                    else:
                        self.fake_H = self.netG(self.var_L, self.variance_map)
                else:
                    if self.w_phase1:
                        self.fake_H = self.netG(self.netG_phase1(self.var_L))
                    else:
                        self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        if self.use_var:
            out_dict['LQ_sigma'] = self.variance_map.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict


    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.loss_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])


    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
