'''
Xiph dataset
support reading images from lmdb, image folder and memcached
'''
import os
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import json
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class Xiph3Mix2Dataset(data.Dataset):
    '''
    Reading the training Xiph dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(Xiph3Mix2Dataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        # self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.LQ_roots = [
            '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_27',
            '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_32',
            '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_37',
        ]

        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        # get metainfo of 60 videos
        with open(self.opt['metainfo_path']) as json_file:
            metainfo_dict = json.load(json_file)
            for key in metainfo_dict.keys():
                metainfo_dict[key]['height'] = int(metainfo_dict[key]['height'])
                metainfo_dict[key]['width'] = int(metainfo_dict[key]['width'])
                metainfo_dict[key]['length'] = int(metainfo_dict[key]['length'])
            self.metainfos = dict(sorted(metainfo_dict.items(), key=lambda item: item[0]))

        # sample frames by resolution
        self.sample_list = []
        min_h = 240
        min_w = 352
        for key, value in self.metainfos.items():
            repeat = int(value['height'] / min_h) * int(value['width'] / min_w)
            self.sample_list += int(value['length'] / 50) * repeat * [str(key)]
            # self.sample_list += [str(key)]

    def __getitem__(self, index):
        index = index % len(self.sample_list)
        LQ_root = self.LQ_roots[random.randint(0, 2)]
        LQ_alpha = random.uniform(0, 1)

        GT_size = self.opt['GT_size']
        video_name = self.sample_list[index]
        video_height = self.metainfos[video_name]['height']
        video_width = self.metainfos[video_name]['width']
        video_length = self.metainfos[video_name]['length']
        center_frame_idx = random.randint(0, video_length - 1)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        N_frames = self.opt['N_frames']
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) >= video_length:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >= video_length) or \
                    (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, video_length - 1)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()

        assert len(neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(len(neighbor_list))

        rnd_h = random.randint(0, max(0, video_height - GT_size)) // 2 * 2
        rnd_w = random.randint(0, max(0, video_width - GT_size)) // 2 * 2
        #### get the GT image (as the center frame)
        img_GT = util.read_yuv_frames(self.GT_root, video_name,
                                      (video_height, video_width),
                                      center_frame_idx, 1,
                                      rnd_h, rnd_w, GT_size)[0]
        #### get LQ images
        img_LQ_l_0 = util.read_yuv_frames(self.LQ_roots[0], video_name,
                                        (video_height, video_width),
                                        center_frame_idx - self.half_N_frames * interval,
                                        N_frames,
                                        rnd_h, rnd_w, GT_size)
        img_LQ_l_2 = util.read_yuv_frames(self.LQ_roots[2], video_name,
                                        (video_height, video_width),
                                        center_frame_idx - self.half_N_frames * interval,
                                        N_frames,
                                        rnd_h, rnd_w, GT_size)
        # img_LQ_l = img_LQ_l_0 * (1 - LQ_alpha) + img_LQ_l_2 * LQ_alpha
        # LQ_alpha = 1
        img_LQ_l = [ i * (1 - LQ_alpha) + j * LQ_alpha for (i, j) in zip(img_LQ_l_0, img_LQ_l_2)]

        # augmentation - flip, rotate
        img_LQ_l.append(img_GT)
        rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
        img_LQ_l = rlt[0:-1]
        img_GT = rlt[-1]
        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()

        # return {'LQs': img_LQs, 'GT': img_GT, 'key': video_name+'_'+str(center_frame_idx)}
        # print(LQ_alpha)
        if N_frames == 1:
            return {'LQ': img_LQs[0], 'GT': img_GT, 'key': video_name+'_'+str(center_frame_idx), 'alpha': LQ_alpha}
        else:
            return {'LQs': img_LQs, 'GT': img_GT, 'key': video_name+'_'+str(center_frame_idx), 'alpha': LQ_alpha}


    def __len__(self):
        # return 10000
        return len(self.sample_list)
