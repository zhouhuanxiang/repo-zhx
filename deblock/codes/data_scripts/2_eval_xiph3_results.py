import os
import os.path as osp
import sys
import argparse
import logging
import json
from collections import OrderedDict
import pandas as pd

sys.path.insert(1, osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util  # noqa: E402
import utils.util


''' base
nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models Xiph_test_all_encoded_37 > ~/zhouhuanxiang/eval_Xiph_test_all_encoded_37 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_metainfos_128.json \
--models Xiph_all_encoded_37 > ~/zhouhuanxiang/eval_Xiph_all_encoded_37 2>&1 &
'''

'''
nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models EDVR_Xiph3_woTSA_M > ~/zhouhuanxiang/eval_EDVR_Xiph3_woTSA_M 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_EDSR_Xiph3 > ~/zhouhuanxiang/eval_test_EDSR_Xiph3 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_CFS_Xiph3 > ~/zhouhuanxiang/eval_test_CFS_Xiph3 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_CFS_Xiph3_25000 > ~/zhouhuanxiang/eval_test_CFS_Xiph3_25000 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_CFS_Xiph3_50000 > ~/zhouhuanxiang/eval_test_CFS_Xiph3_50000 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_CFS_Xiph3_100000 > ~/zhouhuanxiang/eval_test_CFS_Xiph3_100000 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_CFS_2_Xiph3 > ~/zhouhuanxiang/eval_test_CFS_2_Xiph3 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_CFS_3_Xiph3 > ~/zhouhuanxiang/eval_test_CFS_3_Xiph3 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_EDSR_KPN_Xiph3 > ~/zhouhuanxiang/eval_test_EDSR_KPN_Xiph3 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models test_EDSR2_Xiph3 > ~/zhouhuanxiang/eval_EDSR2_Xiph3 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_9_150.json \
--models test_EDSR3_Xiph3 > ~/zhouhuanxiang/eval_EDSR3_Xiph3 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models EDVR_AB_Xiph3_woTSA_M > ~/zhouhuanxiang/eval_EDVR_AB_Xiph3_woTSA_M 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models EDVR_ABC_Xiph3_woTSA_M > ~/zhouhuanxiang/eval_EDVR_ABC_Xiph3_woTSA_M 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models EDVR_C_Xiph3_woTSA_M > ~/zhouhuanxiang/eval_EDVR_C_Xiph3_woTSA_M 2>&1 &

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/2_eval_xiph3_results.py \
--metoinfo_path ~/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json \
--models EDVR_G_Xiph3_woTSA_M > ~/zhouhuanxiang/eval_EDVR_G_Xiph3_woTSA_M 2>&1 &
'''



def parse_args():
    parser = argparse.ArgumentParser(description='eval Xiph results')
    parser.add_argument('--metoinfo_path', type=str, default='', help='metainfo path')
    parser.add_argument('--models', nargs='+', help='eval models')

    args = parser.parse_args()
    return args


def read_frame(result_path, video_name, idx, video_height, video_width, result_format):
    if result_format == 'yuv':
            frame = data_util.read_yuv_frame(result_path, video_name, (video_height, video_width), idx)
    elif result_format == 'png':
        frame_path = os.path.join(result_path, video_name, '{:05d}.png'.format(idx))
        frame = data_util.read_img(None, frame_path)
    else:
        raise NotImplementedError('only support yuv and img format')

    return frame


def sum_ll(ll):
    s = 0.0
    for l in ll:
        s += sum(l)

    return s


def len_ll(ll):
    s = 0
    for l in ll:
        s += len(l)

    return s

def main(args):
    metainfos = {}
    with open(args.metoinfo_path) as json_file:
        metainfo_dict_tmp = json.load(json_file)
        metainfo_dict = {}
        for key in metainfo_dict_tmp.keys():
            for qp in ['27', '32', '37']:
                metainfo_dict[qp + '_' + key] = {}
                metainfo_dict[qp + '_' + key]['height'] = int(metainfo_dict_tmp[key]['height'])
                metainfo_dict[qp + '_' + key]['width'] = int(metainfo_dict_tmp[key]['width'])
                metainfo_dict[qp + '_' + key]['length'] = int(metainfo_dict_tmp[key]['length'])
                metainfo_dict[qp + '_' + key]['yuvname'] = key
        metainfos = dict(sorted(metainfo_dict.items(), key=lambda item: item[0]))

    models = args.models
    for model in models:
        # init
        utils.util.setup_logger('base', '/home/web_server/zhouhuanxiang/disk/log/eval', model,
                                level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')

        if model in ['Xiph_test_all_encoded_37', 'Xiph_test_all_encoded_42']:
            src_path = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test'
            result_path = os.path.join('/home/web_server/zhouhuanxiang/disk/Xiph', model)
            result_format = 'yuv'
        elif model in ['Xiph_all_encoded_37', 'Xiph_all_encoded_42']:
            src_path = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph'
            result_path = os.path.join('/home/web_server/zhouhuanxiang/disk/Xiph', model)
            result_format = 'yuv'
        else:
            src_path = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test'
            result_path = os.path.join('/home/web_server/zhouhuanxiang/disk/log/results', model, 'Xiph_test')
            result_format = 'png'

        # store psnr and ssim in OrderedDict
        eval_results = OrderedDict()
        eval_results['psnr'] = []
        eval_results['ssim'] = []
        eval_results['psnr_y'] = []
        eval_results['ssim_y'] = []
        # travesal of video
        for video_name in metainfos.keys():
            psnr_l = []
            ssim_l = []
            psnr_y_l = []
            ssim_y_l = []
            video_height = metainfos[video_name]['height']
            video_width = metainfos[video_name]['width']
            video_length = metainfos[video_name]['length']
            yuv_name = metainfos[video_name]['yuvname']
            # travesal of frame
            for idx in range(video_length):
                # read frame
                frame_src = read_frame(src_path, yuv_name, idx, video_height, video_width, 'yuv')
                frame_dst = read_frame(result_path, video_name, idx, video_height, video_width, result_format)
                psnr = utils.util.calculate_psnr(frame_src * 255., frame_dst * 255.)
                # ssim = utils.util.calculate_ssim(frame_src * 255., frame_dst * 255.)
                ssim = 0
                #
                # frame_src_y = data_util.bgr2ycbcr(frame_src, only_y=True)
                # frame_dst_y = data_util.bgr2ycbcr(frame_dst, only_y=True)
                # psnr_y = utils.util.calculate_psnr(frame_src_y * 255., frame_dst_y * 255.)
                # ssim_y = utils.util.calculate_ssim(frame_src_y * 255., frame_dst_y * 255.)
                psnr_y = 0
                ssim_y = 0
                logger.info('{:45s} frame {:05d} - PSNR: {:.6f}; SSIM: {:.6f}; PSNR_Y: {:.6f}; SSIM_Y: {:.6f}.'.
                    format(video_name, idx, psnr, ssim, psnr_y, ssim_y))
                #
                psnr_l.append(psnr)
                ssim_l.append(ssim)
                psnr_y_l.append(psnr_y)
                ssim_y_l.append(ssim_y)
            eval_results['psnr'].append(psnr_l)
            eval_results['ssim'].append(ssim_l)
            eval_results['psnr_y'].append(psnr_y_l)
            eval_results['ssim_y'].append(ssim_y_l)

        csv_psnr_dict = {}
        csv_ssim_dict = {}
        csv_ave_dict = {}
        csv_ave_psnr = []
        csv_ave_ssim = []
        csv_video_name = []
        for i, video_name in enumerate(metainfos.keys()):
            ave_psnr = sum(eval_results['psnr'][i]) / len(eval_results['psnr'][i])
            ave_ssim = sum(eval_results['ssim'][i]) / len(eval_results['ssim'][i])
            ave_psnr_y = sum(eval_results['psnr_y'][i]) / len(eval_results['psnr_y'][i])
            ave_ssim_y = sum(eval_results['ssim_y'][i]) / len(eval_results['ssim_y'][i])
            logger.info(
                '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.format(
                    video_name, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))
            csv_psnr_dict[video_name] = eval_results['psnr'][i]
            csv_ssim_dict[video_name] = eval_results['ssim'][i]
            csv_ave_psnr.append(ave_psnr)
            csv_ave_ssim.append(ave_ssim)
            csv_video_name.append(video_name)

        dataframe = pd.DataFrame.from_dict(csv_psnr_dict, orient='index').transpose()
        dataframe.columns = csv_video_name
        dataframe.to_csv(os.path.join('/home/web_server/zhouhuanxiang/disk/log/eval', 'psnr_' + model + '.csv'), index=False, sep=',')
        dataframe = pd.DataFrame.from_dict(csv_ssim_dict, orient='index').transpose()
        dataframe.columns = csv_video_name
        dataframe.to_csv(os.path.join('/home/web_server/zhouhuanxiang/disk/log/eval', 'ssim_' + model + '.csv'), index=False, sep=',')

        # Average PSNR/SSIM results
        ave_psnr = sum_ll(eval_results['psnr']) / len_ll(eval_results['psnr'])
        ave_ssim = sum_ll(eval_results['ssim']) / len_ll(eval_results['ssim'])
        ave_psnr_y = sum_ll(eval_results['psnr_y']) / len_ll(eval_results['psnr_y'])
        ave_ssim_y = sum_ll(eval_results['ssim_y']) / len_ll(eval_results['ssim_y'])
        logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.format(
            model, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))

        csv_ave_psnr.append(ave_psnr)
        csv_ave_ssim.append(ave_ssim)
        csv_video_name.append('average')
        csv_ave_dict['ave_psnr'] = csv_ave_psnr
        csv_ave_dict['ave_ssim'] = csv_ave_ssim

        dataframe = pd.DataFrame(csv_ave_dict, index=csv_video_name)
        dataframe.to_csv(os.path.join('/home/web_server/zhouhuanxiang/disk/log/eval', 'avg_' + model + '.csv'), index=True, sep=',')


if __name__ == '__main__':
    args = parse_args()
    main(args)

    print('done')
