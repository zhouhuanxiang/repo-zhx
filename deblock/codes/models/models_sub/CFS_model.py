import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, ContextualLoss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from models.archs.CFSNet.architecture import CFSNet
from models.loss import GANLoss, GradientPenaltyLoss

logger = logging.getLogger('base')


class CFSNetModel(BaseModel):
    def __init__(self, opt):
        super(CFSNetModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.use_gpu = opt['network_G']['use_gpu']
        self.use_gpu = True
        # 两阶段训练，从训练参数控制
        if self.is_train:
            self.stage = train_opt['stage']
            self.batch_size = opt['datasets']['train']['batch_size']
            if self.stage == 'main_stage' or self.stage == 'tuning_stage':
                self.control_vector = torch.ones(self.batch_size, 512) * train_opt['input_alpha']
        # 结合GAN模型的测试，从测试参数控制
        if not self.is_train and 'input_alpha' in opt['datasets']['test_1'].keys():
            print('input alpha = {}'.format(opt['datasets']['test_1']['input_alpha']))
            self.control_vector = torch.ones(1, 512) * opt['datasets']['test_1']['input_alpha']

        # define network and load pretrained models
        if self.use_gpu:
            self.netG = CFSNet(in_channel=3, out_channel=3,
                                    num_channels=64, num_main_blocks=10,
                                    num_tuning_blocks=10, upscale=None,
                                    task_type='deblock')
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            else:
                self.netG = DataParallel(self.netG)
        else:
            self.netG = networks.define_G(opt)

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

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    if self.stage == 'main_stage':
                        optim_params.append(v)
                    elif self.stage == 'tuning_stage' and 'tuning_blocks' in k:
                        print(k)
                        optim_params.append(v)
                    elif self.stage == 'finetuning_stage':
                        optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            self.use_gan = train_opt['use_gan']
            if self.use_gan:
                self.netD = networks.define_D(opt).to(self.device)  # Discriminator
                self.netD.train()
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

                # D
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D']
                                                    , weight_decay=wd_D, betas=(train_opt['beta1'], 0.999))
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


    def feed_data(self, data, need_GT=True):
        if self.use_gpu:
            self.var_L = data['LQ'].to(self.device)  # LQ
            if need_GT:
                self.real_H = data['GT'].to(self.device)  # GT
        else:
            self.var_L = data['LQ']  # LQ

        # 测试阶段，QP控制
        if not self.is_train and 'QP' in data.keys():
            print(data['QP'])
            if data['QP'] == ['27']:
                self.control_vector = torch.ones(self.var_L.shape[0], 512) * 1.0
            elif data['QP'] == ['32']:
                self.control_vector = torch.ones(self.var_L.shape[0], 512) * 0.5
            elif data['QP'] == ['37']:
                self.control_vector = torch.ones(self.var_L.shape[0], 512) * 0.0
            else:
                print('wrong QP')
        # Mixup训练阶段，mixup参数控制
        if self.is_train and 'alpha' in data.keys():
            self.control_vector = data['alpha'].unsqueeze(1).expand(self.batch_size, 512).float()

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L, self.control_vector)

        if not self.use_gan:
            l_g_total = 0
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            l_g_total += l_pix
            l_g_total.backward()
            self.optimizer_G.step()
            # set log
            self.log_dict['l_pix'] = l_pix.item()
        else:
            loss_total_g = 0
            self.Output = self.fake_H
            self.ground_truth = self.real_H
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
                # gan loss
                pred_g_fake = self.netD(self.Output)
                loss_gan_g = self.loss_gan_w * self.loss_gan(pred_g_fake, True)  # wgan:-E_g(D(x)), to min wasserstein distance:L=E_r(D(x))-E_g(D(x))
                loss_total_g += loss_gan_g

                loss_total_g.backward()
                self.optimizer_G.step()

            # Train D
            self.netD.zero_grad()
            loss_total_d = 0
            pred_d_real = self.netD(self.ground_truth)
            loss_gan_d_real = self.loss_gan(pred_d_real, True)  # wgan:-E_r(D(x))
            pred_d_fake = self.netD(self.Output.detach())
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
                self.log_dict['loss_gan_g'] = loss_gan_g.item()


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L, self.control_vector)
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

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
