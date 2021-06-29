import os.path as osp
import random
import torch
import numpy as np
import torch.utils.data as data
import data.util as util
import cv2


class VimeoLQTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4 (x)
    REDS4 (x)
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VimeoLQTestDataset, self).__init__()
        self.opt = opt
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        self.data_type = self.opt['data_type']
        self.half_N_frames = opt['N_frames'] // 2
        self.LQ_root= opt['dataroot_LQ']
        self.GT_root= opt['dataroot_GT']

        #### determine the LQ frame list
        '''
        N | frames
        1 | 4
        3 | 3,4,5
        5 | 2,3,4,5,6
        7 | 1,2,3,4,5,6,7
        '''
        LQ_frames_list = []
        for i in range(opt['N_frames']):
            LQ_frames_list.append(i + (9 - opt['N_frames']) // 2)
        self.LQ_frames_set = set(LQ_frames_list)

        #### Generate data info and cache data
        self.paths_LQ = util.glob_file_in_file_list(self.LQ_root)
        self.paths_GT = util.glob_file_in_file_list(self.GT_root)

    def __getitem__(self, index):
        scale = self.opt['scale']
        LQ_size = self.opt['LQ_size']

        path_LQ = self.paths_LQ[index]
        path_GT = self.paths_GT[index]
        name_a, name_b = path_LQ.split('/')[-2], path_LQ.split('/')[-1]

        # center LQ
        img_LQ_l = []
        for v in range(7):
            # TODO
            if v == 3:
                img_LQ = util.read_img(None, osp.join(self.GT_root, name_a, name_b, 'im{}.png'.format(v+1)))
                img_LQ_l.append(img_LQ)
            else:
                img_LQ = util.read_img(None, osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(v+1)))
                img_LQ_l.append(img_LQ)

        # LQ_size_tuple = (3, 64, 112) if self.LR_input else (3, 256, 448)


        # TODO
        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        img_LQs = img_LQs[:,:,:,125:381]
        return {'LQs': img_LQs,
                'key': name_a+'_'+name_b,
                }

    def __len__(self):
        return min(len(self.paths_LQ), 1000)
