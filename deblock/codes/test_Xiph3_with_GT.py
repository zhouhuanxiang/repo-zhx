'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import json

import torch.nn as nn
import utils.util as util
import data.util as data_util

'''
python ~/zhouhuanxiang/mmsr/codes/test_Xiph3_with_GT.py

nohup python ~/zhouhuanxiang/mmsr/codes/test_Xiph3_with_GT.py > ~/zhouhuanxiang/test_EDVR_AB_Xiph3_woTSA_M 2>&1 &
'''

def main(test_dataset_folder, test_qp):

    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    #### model
    test_name = 'EDVR_AB_Xiph3_woTSA_M'
    # model_path = '/home/web_server/zhouhuanxiang/disk/log/experiments/EDVR8_semi_C_Xiph37_woTSA_M_1/models/80000_G.pth'
    model_path = '/home/web_server/zhouhuanxiang/disk/log/experiments/'+test_name+'/models/latest_G.pth'
    N_in = 5
    import models.archs.EDVR_AB_arch as EDVR_arch
    model = EDVR_arch.EDVR(64, N_in, 8, 5, 10, predeblur=False, HR_in=True, w_TSA=False)
    #### dataset
    # test_dataset_folder = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded_37'
    GT_dataset_folder = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test'
    metainfo_path = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json'
    # metainfo_path = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_kimono_150.json'

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding_mode = 'replicate'
    
    save_imgs = True
    save_folder = '/home/web_server/zhouhuanxiang/disk/log/results/{}'.format(test_name)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format('Xiph_video_test', test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding_mode))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    #### init Xiph metoinfo
    with open(metainfo_path) as json_file:
        metainfo_dict = json.load(json_file)
        for key in metainfo_dict.keys():
            metainfo_dict[key]['height'] = int(metainfo_dict[key]['height'])
            metainfo_dict[key]['width'] = int(metainfo_dict[key]['width'])
            metainfo_dict[key]['length'] = int(metainfo_dict[key]['length'])
        metainfos = dict(sorted(metainfo_dict.items(), key=lambda item: item[0]))

    for video_name in metainfos.keys():
        video_height = metainfos[video_name]['height']
        video_width = metainfos[video_name]['width']
        video_length = metainfos[video_name]['length']
        # video_length = 10
        # padding
        two_pow = 16
        need_padding_x = (video_width % two_pow != 0)
        need_padding_y = (video_height % two_pow != 0)
        padding_left = 0
        padding_right = 0
        padding_top = 0
        padding_buttom = 0
        if need_padding_x:
            padding_x = two_pow - video_width % two_pow
            padding_left = padding_x // 2
            padding_right = padding_x - padding_left
        if need_padding_y:
            padding_y = two_pow - video_height % two_pow
            padding_top = padding_y // 2
            padding_buttom = padding_y - padding_top
        if need_padding_x or need_padding_y:
            padding = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_buttom), 0)

        save_subfolder = osp.join(save_folder, 'Xiph_test', test_qp + '_' + video_name)
        os.makedirs(save_subfolder, exist_ok=True)

        frame_buffer = {}
        for idx in range(video_length):
            # slide window idx
            select_idx = data_util.index_generation(idx, video_length, N_in, padding=padding_mode)
            print(select_idx)
            # update slide window framebuffer
            for si in select_idx:
                if not si in frame_buffer:
                    img_rgb = data_util.read_yuv_frame(test_dataset_folder, video_name,
                                                       (video_height, video_width), si)
                    img_bgr = img_rgb[:, :, [2, 1, 0]]
                    img_torch = torch.from_numpy(np.ascontiguousarray(np.transpose(img_bgr, (2, 0, 1)))).float().to(device)
                    img_torch = img_torch.unsqueeze(0)
                    # padding
                    if need_padding_x or need_padding_y:
                        img_torch = padding(img_torch)
                    frame_buffer[si] = img_torch
            # del old framebuffer
            if idx - border_frame - 1 in frame_buffer:
                frame_buffer.pop(idx - border_frame - 1)
            # imgs_LQ <- [B N C H W]
            imgs_LQ = frame_buffer[select_idx[0]]
            for si in select_idx[1:]:
                imgs_LQ = torch.cat((imgs_LQ, frame_buffer[si]), 0)
            imgs_LQ = imgs_LQ.unsqueeze(0)
            #
            print(imgs_LQ.shape)
            output = util.single_forward(model, imgs_LQ)
            output = util.tensor2img(output.squeeze(0))
            # padding
            if need_padding_x or need_padding_y:
                output = output[padding_top:padding_top + video_height, padding_left:padding_left + video_width, :]

            img_name = os.path.join(save_subfolder, '{:05d}'.format(idx))
            cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)
            print(osp.join(save_subfolder, '{}.png'.format(img_name)))

'''
    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(data_util.read_img(None, img_GT_path))

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            output = util.single_forward(model, imgs_in)
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)
'''

            # # calculate PSNR
            # output = output / 255.
            # GT = np.copy(img_GT_l[img_idx])
            # # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
            # if data_mode == 'Vid4':  # bgr2y, [0, 1]
            #     GT = data_util.bgr2ycbcr(GT, only_y=True)
            #     output = data_util.bgr2ycbcr(output, only_y=True)
            #
            # output, GT = util.crop_border([output, GT], crop_border)
            # crt_psnr = util.calculate_psnr(output * 255, GT * 255)
            # logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))
            #
            # if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
            #     avg_psnr_center += crt_psnr
            #     N_center += 1
            # else:  # border frames
            #     avg_psnr_border += crt_psnr
            #     N_border += 1

        # avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        # avg_psnr_center = avg_psnr_center / N_center
        # avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        # avg_psnr_l.append(avg_psnr)
        # avg_psnr_center_l.append(avg_psnr_center)
        # avg_psnr_border_l.append(avg_psnr_border)

        # logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
        #             'Center PSNR: {:.6f} dB for {} frames; '
        #             'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
        #                                                            (N_center + N_border),
        #                                                            avg_psnr_center, N_center,
        #                                                            avg_psnr_border, N_border))

    # logger.info('################ Tidy Outputs ################')
    # for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l,
    #                                                           avg_psnr_center_l, avg_psnr_border_l):
    #     logger.info('Folder {} - Average PSNR: {:.6f} dB. '
    #                 'Center PSNR: {:.6f} dB. '
    #                 'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center,
    #                                                  psnr_border))
    # logger.info('################ Final Results ################')
    # logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    # logger.info('Padding mode: {}'.format(padding))
    # logger.info('Model path: {}'.format(model_path))
    # logger.info('Save images: {}'.format(save_imgs))
    # logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
    #             'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
    #                 sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
    #                 sum(avg_psnr_center_l) / len(avg_psnr_center_l),
    #                 sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    LQ_roots = [
        '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded_27',
        '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded_32',
        '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded_37',
    ]

    for LQ_root in LQ_roots:
        LQ_qp = LQ_root.split('/')[-1].split('_')[-1]
        main(LQ_root, LQ_qp)

