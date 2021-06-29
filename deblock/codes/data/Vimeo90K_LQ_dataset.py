import os.path as osp
import random
import torch
import numpy as np
import torch.utils.data as data
import data.util as util
import cv2


class VimeoLQDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4 (x)
    REDS4 (x)
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VimeoLQDataset, self).__init__()
        self.opt = opt
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        self.data_type = self.opt['data_type']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root = opt['dataroot_GT']
        self.LQ_root, self.LLQ_root, self.LHQ_root = opt['dataroot_LQ'], opt['dataroot_LLQ'], opt['dataroot_LHQ']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

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
        self.paths_GT = util.glob_file_in_file_list(self.GT_root)
        self.paths_LQ = util.glob_file_in_file_list(self.LQ_root)
        self.paths_LLQ = util.glob_file_in_file_list(self.LLQ_root)
        self.paths_LHQ = util.glob_file_in_file_list(self.LHQ_root)
        assert len(self.paths_LQ) == len(self.paths_LLQ), 'LQ and LLQ set should have same length'
        assert len(self.paths_LQ) == len(self.paths_LHQ), 'LQ and LHQ set should have same length'

    def __getitem__(self, index):
        scale = self.opt['scale']
        LQ_size = self.opt['LQ_size']
        patch_size = self.opt['patch_size']
        patch_repeat = self.opt['patch_repeat']

        path_LQ = self.paths_LQ[index]
        path_LLQ = self.paths_LLQ[index]
        path_LHQ = self.paths_LHQ[index]
        path_GT = self.paths_GT[index]
        name_a, name_b = path_LQ.split('/')[-2], path_LQ.split('/')[-1]

        #### get LQ images
        indices = random.sample(self.LQ_frames_set, 2)
        # center LQ
        img_LQ_l = []
        img_LQ = util.read_img(None, osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(indices[0])))
        img_LQ_l.append(img_LQ)
        img_LQ = util.read_img(None, osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(indices[1])))
        img_LQ_l.append(img_LQ)
        # LLQ
        img_LLQ = util.read_img(None, osp.join(self.LLQ_root, name_a, name_b, 'im{}.png'.format(indices[1])))
        # LHQ
        img_LHQ = util.read_img(None, osp.join(self.LHQ_root, name_a, name_b, 'im{}.png'.format(indices[1])))
        # GT
        img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b, 'im{}.png'.format(indices[1])))
        img_GT = img_GT - img_LQ_l[1]

        LQ_size_tuple = (3, 64, 112) if self.LR_input else (3, 256, 448)
        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
            img_LLQ = img_LLQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            img_LHQ = img_LHQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            img_GT = img_GT[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_LLQ)
            img_LQ_l.append(img_LHQ)
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:2]
            img_LLQ = rlt[2]
            img_LHQ = rlt[3]
            img_GT = rlt[4]

        # TODO
        # img_LQ_l[0] = img_LQ_l[1].copy()

        patch_labels = []
        patch_offsets = []
        for j in range(patch_repeat):
            rnd_h = random.randint(0, max(0, LQ_size - patch_size))
            rnd_w = random.randint(0, max(0, LQ_size - patch_size))
            rnd_neighbor = random.sample(set([0, 1]), 1)[0]
            # TODO
            # rnd_neighbor = 1
            if rnd_neighbor == 0:
                img_LQ_l[1][rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :] \
                    = img_LLQ[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
                patch_labels.append(0.0)
            else:
                img_LQ_l[1][rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :] \
                    = img_LHQ[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
                patch_labels.append(1.0)
            patch_offsets.append((rnd_h, rnd_w))

            # TODO
            # img_LQ_l[1] = cv2.rectangle(img_LQ_l[1], (rnd_w, rnd_h), (rnd_w + patch_size, rnd_h + patch_size), (0, 0, 255), 1)
            # if type(img_LQ_l[1]) == cv2.UMat:
            #     img_LQ_l[1] = img_LQ_l[1].get()


        # TODO
        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        patch_labels = np.stack(patch_labels, axis=0)
        patch_offsets = np.stack(patch_offsets, axis=0)
        img_GTs = np.expand_dims(img_GT, 0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'Diffs': img_GTs,
                'key': name_a+'_'+name_b,
                'patch_labels': patch_labels,
                'patch_offsets': patch_offsets}

        # TODO
        # # stack LQ images to NHWC, N is the frame number
        # img_LQs = np.stack(img_LQ_l, axis=0)
        # patch_labels = np.stack(patch_labels, axis=0)
        # patch_offsets = np.stack(patch_offsets, axis=0)
        # img_LLQs = np.expand_dims(img_LLQ, 0)
        # img_LHQs = np.expand_dims(img_LHQ, 0)
        # # BGR to RGB, HWC to CHW, numpy to tensor
        # img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        # img_LLQs = img_LLQs[:, :, :, [2, 1, 0]]
        # img_LHQs = img_LHQs[:, :, :, [2, 1, 0]]
        # img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
        #                                                              (0, 3, 1, 2)))).float()
        # img_LLQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LLQs,
        #                                                              (0, 3, 1, 2)))).float()
        # img_LHQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LHQs,
        #                                                              (0, 3, 1, 2)))).float()
        # return {'LQs': img_LQs, 'LLQs': img_LLQs,
        #         'LHQs': img_LHQs, 'key': name_a+'_'+name_b,
        #         'patch_labels': patch_labels,
        #         'patch_offsets': patch_offsets}

    def __len__(self):
        return len(self.paths_LQ)
