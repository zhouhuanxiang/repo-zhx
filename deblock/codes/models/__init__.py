import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'sr_deblur':
        from .SR_deblur_model import SRDeblurModel as M
    elif model == 'sr_mixup':
        from .SR_mixup_model import SRMixupModel as M
    elif model == 'sr_vmaf':
        from .SR_vmaf_model import SRVmafModel as M
    elif model == 'sr_var':
        from .SR_var_model import SRVarModel as M
    elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
        from .SRGAN_model import SRGANModel as M
    elif model == 'srgan_phase2':
        from .SRGAN_phase2_model import SRGANPhase2Model as M
    elif model == 'srgan_phase2_var':
        from .SRGAN_phase2_var_model import SRGANPhase2VarModel as M
    elif model == 'srgan_phase2_mid':
        from .SRGAN_phase2_mid_model import SRGANPhase2MidModel as M
    elif model == 'srgan_phase2_var_ft':
        from .SRGAN_phase2_var_ft_model import SRGANPhase2VarFtModel as M
    elif model == 'cfs':
        from .CFS_model import CFSNetModel as M
    # video restoration
    elif model == 'video_base':
        from .Video_base_model import VideoBaseModel as M
    elif model == 'video_mixup':
        from .Video_mixup_model import VideoMixupModel as M
    elif model == 'video_lstm':
        from .Video_lstm_model import VideoLstmBaseModel as M
    elif model == 'video_semi_A':
        from .Video_semi_model_A import VideoSemiModelA as M
    elif model == 'video_semi_B':
        from .Video_semi_model_B import VideoSemiModelB as M
    elif model == 'video_semi_C':
        from .Video_semi_model_C import VideoSemiModelC as M
    elif model == 'video_base_woDCN':
        from .Video_base_model_woDCN import VideoBaseModel as M
    elif model == 'video_attntion':
        from .Video_attention_model import  VideoAttentionModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
