import numpy as np
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F


class LQVarDataset(data.Dataset):
    '''Read LQ images only in the test phase.'''

    def __init__(self, opt):
        super(LQVarDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.LQ_env = None  # environment for lmdb

        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        # self.paths_LQ = self.paths_LQ[::50]
        assert self.paths_LQ, 'Error: LQ paths are empty.'

        self.win_size = self.opt['win_size']
        self.win_type = self.opt['win_type']
        if self.win_type == 'gauss':
            self.win = self._fspecial_gauss_1d(self.win_size, 1.5)
            self.win = self.win.repeat(1, 1, 1, 1)
        else:
            self.win = torch.tensor([1.0 / self.win_size] * self.win_size)
            self.win = self.win.repeat(1, 1, 1, 1)

    def _init_lmdb(self):
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.LQ_env is None:
            self._init_lmdb()
        LQ_path = None

        # get LQ image
        LQ_path = self.paths_LQ[index]
        resolution = [int(s) for s in self.sizes_LQ[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        H, W, C = img_LQ.shape

        if self.opt['color']:  # change color space if necessary
            img_LQ = util.channel_convert(C, self.opt['color'], [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        img_LQ_sigma = self.calSigma(img_LQ)
        # print(1, img_LQ_sigma.abs().mean())
        img_LQ_sigma = torch.clamp(img_LQ_sigma, 0, 1)
        # print(2, img_LQ_sigma.abs().mean())

        return {'LQ': img_LQ, 'LQ_path': LQ_path, 'LQ_sigma': img_LQ_sigma,
                'key':LQ_path.split('/')[-2]+'_'+LQ_path.split('/')[-1].split('.')[0]}

    def __len__(self):
        return len(self.paths_LQ)

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
        out[:, :, self.win_size // 2:self.win_size // 2 * -1, self.win_size // 2:self.win_size // 2 * -1] = f.unsqueeze(
            0).unsqueeze(0)
        return out

        # N, C, H, W = input.shape
        # out = F.conv2d(input, win, stride=1, padding=(0, self.win_size//2), groups=C)
        # out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=(self.win_size//2, 0), groups=C)
        # return out