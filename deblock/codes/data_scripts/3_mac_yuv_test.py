import os
import os.path as osp
import sys
import argparse
import logging
import json
from collections import OrderedDict
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(1, osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util  # noqa: E402
import utils.util


'''
python ~/zhouhuanxiang/mmsr/codes/data_scripts/3_mac_yuv_test.py

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/3_mac_yuv_test.py > ~/zhouhuanxiang/0 2>&1 &

'''

def _fspecial_gauss_1d(win_size, sigma):
    coords = torch.arange(win_size).to(dtype=torch.float)
    coords -= win_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win, win_size):
    N, C, H, W = input.shape
    f = F.conv2d(input, win, stride=1, groups=C)
    f = F.conv2d(f, win.transpose(2, 3), stride=1, groups=C)

    out = torch.zeros(input.shape)
    out[:, :, win_size//2:win_size//2 * -1, win_size//2:win_size//2 * -1] = f.unsqueeze(0).unsqueeze(0)
    return out

def gaussian_filter_numpy(r, win, win_size):
    r = torch.from_numpy(np.ascontiguousarray(r)).float().unsqueeze(0).unsqueeze(0)
    r = gaussian_filter(r, win, win_size)
    r = utils.util.tensor2img(r.squeeze(0))
    return r

def main():
    if 0:
        yuv_path_original = '/Users/zhx/Desktop/Xiph-test-yuv'
        yuv_path_compress = '/Users/zhx/Desktop/Xiph_test_all_encoded_37'
        img_path = '/Users/zhx/Desktop/Xiph_img'
        yuv_metainfo_path = '/Users/zhx/Desktop/repo/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json'
    else:
        yuv_path_original = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test'
        yuv_path_compress = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded_37'
        img_path = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded_37_PQB'
        yuv_metainfo_path = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos_10_150.json'

        # yuv_path_original = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph'
        # yuv_path_compress = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_37'
        # img_path = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_37_PQB'
        # yuv_metainfo_path = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_metainfos_128.json'

    metainfos = {}
    with open(yuv_metainfo_path) as json_file:
        metainfo_dict = json.load(json_file)
        for key in metainfo_dict.keys():
            metainfo_dict[key]['height'] = int(metainfo_dict[key]['height'])
            metainfo_dict[key]['width'] = int(metainfo_dict[key]['width'])
            metainfo_dict[key]['length'] = int(metainfo_dict[key]['length'])
        metainfos = dict(sorted(metainfo_dict.items(), key=lambda item: item[0]))
    print(metainfos)

    for i, video_name in enumerate(metainfos.keys()):
        # if i % 6 != 5:
        #     continue

        if video_name != 'BasketballPass_416x240_50':
            continue

        video_height = metainfos[video_name]['height']
        video_width = metainfos[video_name]['width']
        video_length = metainfos[video_name]['length']

        win_size = int(video_height / 70) // 2 * 2 + 1
        win = _fspecial_gauss_1d(win_size, win_size / 3.0)
        win = win.repeat(1, 1, 1, 1)

        os.makedirs(os.path.join(img_path, video_name), exist_ok=True)

        gt_1 = data_util.read_yuv_frame(yuv_path_original, video_name, (video_height, video_width), 0).astype('float')
        im_1 = data_util.read_yuv_frame(yuv_path_compress, video_name, (video_height, video_width), 0).astype('float')
        # r_1 = gaussian_filter_numpy(np.abs(im_1 - gt_1).mean(2), win, win_size)
        gt_2 = data_util.read_yuv_frame(yuv_path_original, video_name, (video_height, video_width), 0).astype('float')
        im_2 = data_util.read_yuv_frame(yuv_path_compress, video_name, (video_height, video_width), 0).astype('float')
        # r_2 = gaussian_filter_numpy(np.abs(im_2 - gt_2).mean(2), win, win_size)
        for idx in range(0, video_length):
            print('{} {:05d}/{:05d} {}'.format(video_name, idx, video_length, win_size))
            gt_0 = gt_1
            im_0 = im_1
            # r_0 = r_1
            gt_1 = gt_2
            im_1 = im_2
            # r_1 = r_2
            gt_2 = data_util.read_yuv_frame(yuv_path_original, video_name, (video_height, video_width), min(idx + 1, video_length - 1)).astype('float')
            im_2 = data_util.read_yuv_frame(yuv_path_compress, video_name, (video_height, video_width), min(idx + 1, video_length - 1)).astype('float')
            # r_2 = gaussian_filter_numpy(np.abs(im_2 - gt_2).mean(2), win, win_size)

            # if idx == 0:
            #     PQB_map = (r_1 > r_2).astype('float') * 255.0
            # elif idx == video_length - 1:
            #     PQB_map = (r_1 > r_0).astype('float') * 255.0
            # else:
            #     PQB_map = np.logical_and(r_1 > r_0, r_1 > r_2).astype('float') * 255.0

            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            # PQB_map = cv2.erode(PQB_map, kernel)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # PQB_map = cv2.dilate(PQB_map, kernel)


            # cv2.imwrite(os.path.join(img_path, video_name, '{:05d}.png'.format(idx)), PQB_map)

            # cv2.imwrite(os.path.join(img_path, video_name, '{:05d}.png'.format(idx)), np.expand_dims(PQB_map, axis=2) * im_1)

            cv2.imwrite(os.path.join(img_path, video_name, '{:05d}_LQ.png'.format(idx)), im_1 * 255.0)
            # cv2.imwrite(os.path.join(img_path, video_name, '{:05d}_HQ.png'.format(idx)), gt_1 * 255.0)


if __name__ == '__main__':
    main()
    print('done')
