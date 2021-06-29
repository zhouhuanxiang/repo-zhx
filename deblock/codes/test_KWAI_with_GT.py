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
import time

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch

'''
python ~/zhouhuanxiang/mmsr/codes/test_KWAI_with_GT.py
'''

def main():

    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    #### dataset
    test_dataset_folder = '/home/web_server/zhouhuanxiang/disk/data/HD_UGC_crf35_raw_test'
    #### model
    test_name = 'EDVR_KWAI35_woTSA_M'
    model_path = '/home/web_server/zhouhuanxiang/disk/log/experiments/EDVR_KWAI35_woTSA_M/models/latest_G.pth'
    N_in = 5
    model = EDVR_arch.EDVR(64, N_in, 8, 5, 10, predeblur=False, HR_in=True, w_TSA=False)
    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'replicate'

    save_imgs = True
    save_folder = '/home/web_server/zhouhuanxiang/disk/log/results/{}'.format(test_name)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(test_name, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    # for each subfolder
    for subfolder in subfolder_l:
        print(subfolder)

        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            print(img_path, select_idx)

            # without feature
            output = util.single_forward(model, imgs_in)

            # # save feature 
            # output, aligned_fea = util.single_forward_all_results(model, imgs_in)
            # aligned_fea = aligned_fea.squeeze(0)
            # for i in range(aligned_fea.shape[0]):
            #     fea = util.tensor2img(aligned_fea[i])
            #     print(osp.join(save_subfolder, '{}_{}.png'.format(img_name, i)))
            #     cv2.imwrite(osp.join(save_subfolder, '{}_{}.png'.format(img_name, i)), fea)

            # # time test
            # time1 = time.time()
            # for i in range(50):
            #     output = util.single_forward(model, imgs_in)
            # time2 = time.time()
            # print(1 / (time2 - time1) * 50)
            
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # calculate PSNR
            # output = output / 255.
            # GT = np.copy(img_GT_l[img_idx])
            # # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
            # if data_mode == 'Vid4':  # bgr2y, [0, 1]
            #     GT = data_util.bgr2ycbcr(GT, only_y=True)
            #     output = data_util.bgr2ycbcr(output, only_y=True)

            # output, GT = util.crop_border([output, GT], crop_border)
            # crt_psnr = util.calculate_psnr(output * 255, GT * 255)
            # logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))

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
    # logger.info('Flip test: {}'.format(flip_test))
    # logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
    #             'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
    #                 sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
    #                 sum(avg_psnr_center_l) / len(avg_psnr_center_l),
    #                 sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
