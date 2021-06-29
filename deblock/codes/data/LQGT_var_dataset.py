import random
import numpy as np
import cv2
import lmdb
import torch
import torch.nn.functional as F
import torch.utils.data as data
import data.util as util


class LQGTVarDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(LQGTVarDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        if self.opt['phase'] == 'val' :
            self.paths_GT = self.paths_GT[500:3500:150]
            self.paths_LQ = self.paths_LQ[500:3500:150]   
            # self.paths_GT = self.paths_GT[:10]
            # self.paths_LQ = self.paths_LQ[:10]
            self.sizes_GT = len(self.paths_GT)
            self.sizes_LQ = len(self.paths_LQ)
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

        self.win_size = self.opt['win_size']
        self.win_type = self.opt['win_type']
        if self.win_type == 'gauss':
            self.win = self._fspecial_gauss_1d(self.win_size, 1.5)
            self.win = self.win.repeat(1, 1, 1, 1)
        else:
            self.win = torch.tensor([1.0 / self.win_size] * self.win_size)
            self.win = self.win.repeat(1, 1, 1, 1)

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        if self.opt['color']:  # change color space if necessary
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            resolution = [int(s) for s in self.sizes_LQ[index].split('_')
                          ] if self.data_type == 'lmdb' else None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(img_GT, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        if self.opt['color']:  # change color space if necessary
            img_LQ = util.channel_convert(C, self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        img_LQ_sigma = self.calSigma(img_LQ)
        # print(1, img_LQ_sigma.abs().mean())
        img_LQ_sigma = torch.clamp(img_LQ_sigma, 0, 1)
        # print(2, img_LQ_sigma.abs().mean())

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path, 'LQ_sigma': img_LQ_sigma,
                'key':GT_path.split('/')[-2]+'_'+GT_path.split('/')[-1].split('.')[0]}

    def __len__(self):
        return len(self.paths_GT)

    def calSigma(self, img):
        img = img.mean(0, keepdim=True).unsqueeze(0)
        img_sq = img.pow(2)
        img_mean = self.gaussian_filter(img, self.win, self.win_size)
        img_sq_mean = self.gaussian_filter(img_sq, self.win, self.win_size)
        img_sigma_sq = img_sq_mean - img_mean.pow(2)

        # img_sigma = img_sigma_sq.sqrt()
        img_sigma = img_sigma_sq

        img_sigma = img_sigma.squeeze(0)

        return img_sigma * 30


    def _fspecial_gauss_1d(self, size, sigma):
        r"""Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        return g.unsqueeze(0).unsqueeze(0)

    def gaussian_filter(self, input, win, size):
        r""" Blur input with 1-D kernel
        Args:
            input (torch.Tensor): a batch of tensors to be blured
            window (torch.Tensor): 1-D gauss kernel

        Returns:
            torch.Tensor: blured tensors
        """

        N, C, H, W = input.shape
        f = F.conv2d(input, win, stride=1, groups=C)
        f = F.conv2d(f, win.transpose(2, 3), stride=1, groups=C)

        out = torch.zeros(input.shape)
        out[:, :, self.win_size//2:self.win_size//2 * -1, self.win_size//2:self.win_size//2 * -1] = f.unsqueeze(0).unsqueeze(0)
        return out

        # N, C, H, W = input.shape
        # out = F.conv2d(input, win, stride=1, padding=(0, self.win_size//2), groups=C)
        # out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=(self.win_size//2, 0), groups=C)
        # return out