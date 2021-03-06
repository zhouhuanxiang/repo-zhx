import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

# from mmedit.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a editor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[250, 250],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    # args = parse_args()
    #
    # if len(args.shape) == 1:
    #     input_shape = (3, args.shape[0], args.shape[0])
    # elif len(args.shape) == 2:
    #     input_shape = (3, ) + tuple(args.shape)
    # elif len(args.shape) == 3:
    #     input_shape = tuple(args.shape)
    # else:
    #     raise ValueError('invalid input shape')

    # cfg = Config.fromfile(args.config)
    # model = build_model(
    #     cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    # model.eval()

    # import models.archs.SRResNet_o2m_ft_arch as SRResNet_o2m_ft_arch
    # model = SRResNet_o2m_ft_arch.MSRResNet(in_nc=3, out_nc=3, nf=64, nb=16, upscale=1)

    import models.deblur_archs.uvud_arch as uvud_arch
    model = uvud_arch.UDVDPlus()

    input_shape = (3, 720, 1088)

    # if hasattr(model, 'forward_dummy'):
    #     model.forward = model.forward_dummy
    # else:
    #     raise NotImplementedError(
    #         'FLOPs counter is currently not currently supported '
    #         f'with {model.__class__.__name__}')

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()

'''
python ~/zhouhuanxiang/mmsr/codes/get_flops.py
'''