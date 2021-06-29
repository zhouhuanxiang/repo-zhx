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


class XiphTestDataset(data.Dataset):
    '''
    Reading the training Xiph dataset
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    only support single frame
    '''

    def __init__(self, opt):
        super(XiphTestDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.LQ_root = opt['dataroot_LQ']

        self.LR_input = False  # low resolution inputs

        # get metainfo of 60 videos
        with open(self.opt['metainfo_path']) as json_file:
            metainfo_dict = json.load(json_file)
            for key in metainfo_dict.keys():
                metainfo_dict[key]['height'] = int(metainfo_dict[key]['height'])
                metainfo_dict[key]['width'] = int(metainfo_dict[key]['width'])
                metainfo_dict[key]['length'] = int(metainfo_dict[key]['length'])
            self.metainfos = dict(sorted(metainfo_dict.items(), key=lambda item: item[0]))

        # sample frames by resolution
        self.sample_list = self.metainfos.items()
        self.sample_list = []
        for key in self.metainfos.keys():
            key_l = [key] * self.metainfos[key]['length']
            idx_l = list(range(self.metainfos[key]['length']))
            self.sample_list.extend(list(zip(key_l, idx_l)))

    def __getitem__(self, index):
        index = index % len(self.sample_list)

        GT_size = self.opt['GT_size']
        video_name = self.sample_list[index][0]
        video_height = self.metainfos[video_name]['height']
        video_width = self.metainfos[video_name]['width']
        center_frame_idx = self.sample_list[index][1]

        #### get LQ image
        img_LQ = util.read_yuv_frame(self.LQ_root, video_name,
                                        (video_height, video_width),
                                        center_frame_idx)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        LQ_path = '{}/{:05d}.png'.format(video_name, center_frame_idx)
        # print(img_LQ.shape, LQ_path)

        return {'LQ': img_LQ, 'LQ_path': LQ_path}


    def __len__(self):
        return len(self.sample_list)
