import torch
import torch.nn as nn
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.discriminator_patch_arch as SRGAN_patch_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.ProxIQANet_arch as ProxIQANet_arch
# import models.archs.EDVR_arch as EDVR_arch
import functools

from repo.MGANet.MGANet import Gen_Guided_UNet
# import models.convlstm.Sakuya_arch as Sakuya_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        import models.archs.SRResNet_arch as SRResNet_arch
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_mid':
        import models.archs.SRResNet_arch_mid as SRResNet_arch
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet2':
        import models.archs.SRResNet2_arch as SRResNet2_arch
        netG = SRResNet2_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet3':
        import models.archs.SRResNet3_arch as SRResNet3_arch
        netG = SRResNet3_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_GDN':
        import models.archs.SRResNet_GDN_arch as SRResNet_GDN_arch
        netG = SRResNet_GDN_arch.MSRResNetGDN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_KPN':
        import models.archs.SRResNet_KPN_arch as SRResNet_KPN_arch
        netG = SRResNet_KPN_arch.MSRResNetKPN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_cbam':
        import models.archs.SRResNet_cbam_arch as SRResNet_cbam_arch
        netG = SRResNet_cbam_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_o2m':
        import models.archs.SRResNet_o2m_arch as SRResNet_arch
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_sa':
        import models.archs.SRResNet_sa_arch as SRResNet_arch
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_o2m_spectral':
        import models.archs.SRResNet_o2m_spectral_arch as SRResNet_arch
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_o2m_var':
        import models.archs.SRResNet_o2m_var_arch as SRResNet_arch
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_o2m_ft':
        import models.archs.SRResNet_o2m_ft_arch as SRResNet_arch
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    # video restoration
    elif which_model == 'n3net':
        import models.archs.n3net.n3net as n3net
        netG = n3net.N3NetVideo(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              front_RBs=opt_net['front_RBs'], back_RBs=opt_net['back_RBs'])
    elif which_model == 'EDVR':
        import models.archs.EDVR_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR_KPN':
        import models.archs.EDVR_KPN_arch as EDVR_arch
        netG = EDVR_arch.EDVR_KPN(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR1':
        import models.archs.EDVR1_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR2':
        import models.archs.EDVR2_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR3':
        import models.archs.EDVR3_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'],
                              offset_only=opt_net['offset_only'])
    elif which_model == 'EDVR4':
        import models.archs.EDVR4_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'],
                              return_offset=opt_net['return_offset'])
    elif which_model == 'EDVR5':
        import models.archs.EDVR5_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR6':
        import models.archs.EDVR6_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR7':
        import models.archs.EDVR7_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR8':
        import models.archs.EDVR8_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR9':
        import models.archs.EDVR9_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR10':
        import models.archs.EDVR10_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-G':
        import models.archs.EDVR_G_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-AB':
        import models.archs.EDVR_AB_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-ABG':
        import models.archs.EDVR_ABG_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar1':
        import models.archs.EDVR_haar1_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar2':
        import models.archs.EDVR_haar2_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar3':
        import models.archs.EDVR_haar3_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar5':
        import models.archs.EDVR_haar5_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar6':
        import models.archs.EDVR_haar6_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar7':
        import models.archs.EDVR_haar7_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar8':
        import models.archs.EDVR_haar8_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar9':
        import models.archs.EDVR_haar9_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar10':
        import models.archs.EDVR_haar10_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar11':
        import models.archs.EDVR_haar11_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar51':
        import models.archs.EDVR_haar51_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar31':
        import models.archs.EDVR_haar31_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar13':
        import models.archs.EDVR_haar13_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'EDVR-haar15':
        import models.archs.EDVR_haar15_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'TOF':
        import models.archs.TOF_arch as TOF_arch
        netG = TOF_arch.TOFlow()
    elif which_model == 'TOF_Pwc':
        import models.archs.TOF_Pwc_arch as TOF_Pwc_arch
        netG = TOF_Pwc_arch.TOFPwclow()
    elif which_model == 'LunaTokis':
        netG = Sakuya_arch.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                     groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                     back_RBs=opt_net['back_RBs'])
            # elif which_model == 'EDVR_woDCN':
    #     import models.archs.EDVR_woDCN_arch as EDVR_arch
    #     netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
    #                           groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
    #                           back_RBs=opt_net['back_RBs'], center=opt_net['center'],
    #                           predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
    #                           w_TSA=opt_net['w_TSA'], w_GCB=opt_net['w_GCB'])
    elif which_model == 'MGANet':
        netG = Gen_Guided_UNet(input_size=opt_net['input_size'])
    elif which_model == 'Unet':
        import repo.CycleGAN.networks as unet_networks
        netG = unet_networks.define_G(2 * 3, 1, opt_net['nf'], 
                                      opt_net['G_type'], opt_net['norm'],
                                      opt_net['dropout'], opt_net['init_type'], 
                                      opt_net['init_gain'])
    elif which_model == 'uvud':
        import models.deblur_archs.uvud_arch as uvud_arch
        netG = uvud_arch.UDVDPlus(opt_net['blurnet_nf'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'spectral':
        norm_layer = functools.partial(nn.utils.spectral_norm)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_patch' and opt_net['Instance'] == 'spectral':
        netD = SRGAN_patch_arch.NLayerDiscriminatorSpectral(input_nc=opt_net['in_nc'], ndf=opt_net['nf'], n_layers=3)
    elif which_model == 'discriminator_patch':
        norm_layer = get_norm_layer(norm_type=opt_net['Instance'])
        netD = SRGAN_patch_arch.NLayerDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'], n_layers=3, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

def define_I(opt):
    opt_net = opt['network_I']
    which_model = opt_net['which_model_I']
    use_sigmoid = opt_net['use_sigmoid']


    net_I= ProxIQANet_arch.ProxIQANet(use_sigmoid)

    return net_I