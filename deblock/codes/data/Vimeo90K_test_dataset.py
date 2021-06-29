import os.path as osp
import random
import torch
import numpy as np
import torch.utils.data as data
import data.util as util



class VimeoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4 (x)
    REDS4 (x)
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VimeoTestDataset, self).__init__()
        self.opt = opt
        self.all_gt = opt['all_gt']
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        self.data_type = self.opt['data_type']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        #### determine the LQ frame list
        '''
        N | frames
        1 | 4
        3 | 3,4,5
        5 | 2,3,4,5,6
        7 | 1,2,3,4,5,6,7
        '''
        self.LQ_frames_list = []
        for i in range(opt['N_frames']):
            self.LQ_frames_list.append(i + (9 - opt['N_frames']) // 2)

        #### Generate data info and cache data
        if opt['name'].lower() in ['vimeo90k-test']:
            subfolders_LQ = util.glob_file_in_file_list(self.LQ_root)
            subfolders_GT = util.glob_file_in_file_list(self.GT_root)
            self.paths_LQ = subfolders_LQ
            self.paths_GT = subfolders_GT
            assert len(self.paths_GT) == len(self.paths_LQ), 'GT and LQ set should have same length'
        else:
            raise ValueError(
                'Not support video test dataset. Support Vimeo90k-Test.')

    def __getitem__(self, index):
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        path_GT = self.paths_GT[index]
        path_LQ = self.paths_LQ[index]
        name_a, name_b = path_GT.split('/')[-2], path_GT.split('/')[-1]
        #### get the GT image (as the center frame or all frames)
        img_GT_l = []
        if self.all_gt:
            for v in self.LQ_frames_list:
                img_GT = util.read_img(None,
                                       osp.join(path_GT, 'im{}.png'.format(v)))
                img_GT_l.append(img_GT)
        else:
            img_GT = util.read_img(None, osp.join(path_GT, 'im4.png'))
            img_GT_l.append(img_GT)
        #### get LQ images
        LQ_size_tuple = (3, 64, 112) if self.LR_input else (3, 256, 448)
        img_LQ_l = []
        for v in self.LQ_frames_list:
            img_LQ = util.read_img(None,
                                   osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(v)))
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                # img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
                img_GT_l = [v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in img_GT_l]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                # img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
                img_GT_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_GT_l]

            # augmentation - flip, rotate
            img_LQ_l.extend(img_GT_l)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            if self.all_gt:
                img_LQ_l = rlt[0:len(img_GT_l)]
                img_GT_l = rlt[-1 * len(img_LQ_l):]
            else:
                img_LQ_l = rlt[0:-1]
                img_GT = rlt[-1:]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GTs = np.stack(img_GT_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs,
                                                                     (0, 3, 1, 2)))).float()

        return {'LQs': img_LQs, 'GTs': img_GTs, 'key': name_a+'_'+name_b}

    def __len__(self):
        return len(self.paths_GT)
