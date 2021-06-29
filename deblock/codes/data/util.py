import os
import math
import pickle
import random
import numpy as np
import glob
import torch
import cv2
import glob
import sys
import os.path as osp
# sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
# import utils.util as util
import re
import multiprocessing

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
            # if is_image_file(fnamesss ) and fname[-9:] == '00001.png':
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    if not images:
        images = sorted(images)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    """get image path list
    support lmdb or image files"""
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths, sizes


def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))


def glob_file_in_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*', '*')))


###################### read images ######################
def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_img_seq(path):
    """Read a sequence of images from a given folder path
    Args:
        path (list/str): list of image paths/image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))
    img_l = [read_img(None, v) for v in img_path_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def index_generation(crt_i, max_n, N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l


####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def augment_flow(img_list, flow_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees) with flows"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = flow[:, ::-1, :]
            flow[:, :, 0] *= -1
        if vflip:
            flow = flow[::-1, :, :]
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]

    return rlt_img_list, rlt_flow_list


def channel_convert(in_c, tar_type, img_list):
    """conversion among BGR, gray and y"""
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    """same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    """bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    """img_in: Numpy, HWC or HW"""
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


####################
# Functions
####################


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()


# def run_vmaf(img_LQ, img_GT, filename):
#     # 0) get process pid
#     pid = os.getpid()
#     filename = str(pid) + '_' + filename
#     # 1) save image as .png
#     util.save_img(img_LQ * 255.0, '/home/web_server/zhouhuanxiang/disk/tmp/{}_LQ.png'.format(filename))
#     util.save_img(img_GT * 255.0, '/home/web_server/zhouhuanxiang/disk/tmp/{}_GT.png'.format(filename))
#     # 2) run ffmpeg with vmaf filter to get the vmaf score
#     output = os.popen('ffmpeg '\
#                       '-i /home/web_server/zhouhuanxiang/disk/tmp/{}_LQ.png '\
#                       '-i /home/web_server/zhouhuanxiang/disk/tmp/{}_GT.png '\
#                       '-lavfi '\
#                       'libvmaf="model_path=/home/web_server/zhouhuanxiang/disk/vmaf/model/vmaf_v0.6.1.pkl"'\
#                       ' -hide_banner -f null - 2>&1'.format(filename, filename)).read()
#     vmaf = (re.compile("score: (\d+.\d+)").findall(output))[0]
#     vmaf = float(vmaf) / 100.0
#
#     os.system('rm /home/web_server/zhouhuanxiang/disk/tmp/{}_LQ.png '\
#               '/home/web_server/zhouhuanxiang/disk/tmp/{}_GT.png'.format(filename, filename))
#
#     # print(vmafs)
#     return vmafs


# def run_vmaf_pytorch(fake_H, real_H):
#     img_LQ = fake_H.float().cpu()
#     img_HQ = real_H.float().cpu()
#
#     pid = os.getpid()
#     B, C, H, W = img_HQ.shape
#     vmafs = torch.zeros(B)
#
#     for i in range(B):
#         # 1) save the image as .png
#         var_L_i = img_LQ[i]
#         real_H_i = img_HQ[i]
#         img_L_i = util.tensor2img(var_L_i)
#         img_H_i = util.tensor2img(real_H_i)
#         util.save_img(img_L_i, '/home/web_server/zhouhuanxiang/disk/tmp/LQ{}_{}.png'.format(i, pid))
#         util.save_img(img_H_i, '/home/web_server/zhouhuanxiang/disk/tmp/HQ{}_{}.png'.format(i, pid))
#         # 2) run ffmpeg with vmaf filter to get the vmaf score
#         output = os.popen('ffmpeg '\
#                           '-i /home/web_server/zhouhuanxiang/disk/tmp/LQ{}_{}.png '\
#                           '-i /home/web_server/zhouhuanxiang/disk/tmp/HQ{}_{}.png '\
#                           '-lavfi '\
#                           'libvmaf="model_path=/home/web_server/zhouhuanxiang/disk/vmaf/model/vmaf_v0.6.1.pkl"'\
#                           ' -hide_banner -f null - 2>&1'.format(i, pid, i, pid)).read()
#         vmaf = (re.compile("score: (\d+.\d+)").findall(output))[0]
#         vmafs[i] = float(vmaf) / 100.0
#
#     # for i in range(B):
#     #     os.system('rm /home/web_server/zhouhuanxiang/disk/tmp/LQ{}_{}.png'.format(i, pid))
#     #     os.system('rm /home/web_server/zhouhuanxiang/disk/tmp/HQ{}_{}.png'.format(i, pid))
#
#     return vmafs
#
#         # # 2) convert .png to .yuv use ffmpeg
#         # os.system('ffmpeg -i /home/web_server/zhouhuanxiang/disk/tmp/LQ{}.png '\
#         #           '-f rawvideo -pix_fmt yuv420p -y -hide_banner -loglevel panic '\
#         #           '/home/web_server/zhouhuanxiang/disk/tmp/LQ{}.yuv'.format(i, i))
#         # os.system('ffmpeg -i /home/web_server/zhouhuanxiang/disk/tmp/HQ{}.png ' \
#         #           '-f rawvideo -pix_fmt yuv420p -y -hide_banner -loglevel panic ' \
#         #           '/home/web_server/zhouhuanxiang/disk/tmp/HQ{}.yuv'.format(i, i))
#         # # 3) run vmafossexec to get the vmaf score
#         # os.system('vmafossexec yuv420p {} {} '\
#         #           '/home/web_server/zhouhuanxiang/disk/tmp/HQ{}.yuv '\
#         #           '/home/web_server/zhouhuanxiang/disk/tmp/LQ{}.yuv '\
#         #           '/home/web_server/zhouhuanxiang/disk/vmaf/model/vmaf_v0.6.1.pkl '\
#         #           '--log /home/web_server/zhouhuanxiang/disk/tmp/vmaf{}.xml'.format(H, W, i, i, i))
#         # # 4) read vmaf socre from xml
#         # xml_i = minidom.parse('/home/web_server/zhouhuanxiang/disk/tmp/vmaf{}.xml'.format(i))
#         # item = xml_i.getElementsByTagName('frame')[0]
#         # vmafs[i, 0] = float(item.attributes['vmaf'].value) / 100.0


# def run_vmaf_pytorch_parallel_worker(pid, i, var_L_i, real_H_i, vmafs):
#     # 1) save the image as .png
#     img_L_i = util.tensor2img(var_L_i)
#     img_H_i = util.tensor2img(real_H_i)
#     util.save_img(img_L_i, '/home/web_server/zhouhuanxiang/disk/tmp/LQ{}_{}.png'.format(i, pid))
#     util.save_img(img_H_i, '/home/web_server/zhouhuanxiang/disk/tmp/HQ{}_{}.png'.format(i, pid))
#     # 2) run ffmpeg with vmaf filter to get the vmaf score
#     output = os.popen('ffmpeg ' \
#                       '-i /home/web_server/zhouhuanxiang/disk/tmp/LQ{}_{}.png ' \
#                       '-i /home/web_server/zhouhuanxiang/disk/tmp/HQ{}_{}.png ' \
#                       '-lavfi ' \
#                       'libvmaf="model_path=/home/web_server/zhouhuanxiang/disk/vmaf/model/vmaf_v0.6.1.pkl"' \
#                       ' -hide_banner -f null - 2>&1'.format(i, pid, i, pid)).read()
#     vmaf = (re.compile("score: (\d+.\d+)").findall(output))[0]
#     vmafs[i] = float(vmaf) / 100.0


# def run_vmaf_pytorch_parallel(fake_H, real_H):
#     img_LQ = fake_H.detach().float().cpu()
#     img_HQ = real_H.detach().float().cpu()
#
#     B = img_HQ.shape[0]
#     pid = os.getpid()
#
#     manager = multiprocessing.Manager()
#     return_dict = manager.dict()
#     jobs = []
#
#     for i in range(B):
#         var_L_i = img_LQ[i]
#         var_H_i = img_HQ[i]
#         p = multiprocessing.Process(target=run_vmaf_pytorch_parallel_worker, args=(pid, i, var_L_i, var_H_i, return_dict))
#         jobs.append(p)
#         p.start()
#
#     for proc in jobs:
#         proc.join()
#
#     # sort return_dict by key
#     return_dict = dict(sorted(return_dict.items(), key=lambda item: item[0]))
#     return_dict = np.array(list(return_dict.values())).astype(float)
#     # print(return_dict)
#
#     vmafs = torch.tensor(return_dict, dtype=torch.float)
#     return vmafs


def YUV2RGB(yuv):
    # m = np.array([[ 1.0, 1.0, 1.0],
    #              [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
    #              [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])

    # rgb = np.dot(yuv,m)
    # rgb[:,:,0]-=179.45477266423404
    # rgb[:,:,1]+=135.45870971679688
    # rgb[:,:,2]-=226.8183044444304
    # rgb = rgb.clip(0,255)
    # rgb = rgb[:, :, (2, 1, 0)]

    m = np.array([[1.0, 1.0, 1.0],
                  [0, -0.3455, 1.7790],
                  [1.4075, -0.7169, 0]])

    yuv = yuv.astype('float')
    yuv[:, :, 1] -= 128
    yuv[:, :, 2] -= 128
    rgb = np.dot(yuv, m)
    rgb = rgb.clip(0, 255)
    rgb = rgb[:, :, (2, 1, 0)]
    return rgb


def read_yuv_frames(video_path, video_name, dims, startfrm, numfrm, rnd_h, rnd_w, patch_size):
    fp = open(os.path.join(video_path, video_name+'.yuv'), 'rb')
    blk_size = np.prod(dims) * 3 / 2
    fp.seek(np.int(blk_size * startfrm), 0)
    d00 = dims[0] // 2
    d01 = dims[1] // 2

    imgs = []
    for i in range(numfrm):
        # Y channel
        y = fp.read(dims[0] * dims[1])
        y = np.frombuffer(y, dtype=np.uint8)
        y = y.reshape(dims)
        y = y[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size]
        # U channel
        u = fp.read(d00 * d01)
        u = np.frombuffer(u, dtype=np.uint8)
        u = u.reshape(d00, d01)
        u = u[rnd_h // 2:(rnd_h + patch_size) // 2, rnd_w // 2:(rnd_w + patch_size) // 2]
        # V channel
        v = fp.read(d00 * d01)
        v = np.frombuffer(v, dtype=np.uint8)
        v = v.reshape(d00, d01)
        v = v[rnd_h // 2:(rnd_h + patch_size) // 2, rnd_w // 2:(rnd_w + patch_size) // 2]
        # scale UV channel
        enlarge_u = cv2.resize(u, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        enlarge_v = cv2.resize(v, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # to YUV channels to BGR channels
        img_yuv = cv2.merge([y, enlarge_u, enlarge_v])
        # img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) / 255.0
        img_rgb = YUV2RGB(img_yuv) / 255.0
        imgs.append(img_rgb)
    fp.close()
    return imgs


def read_yuv_frame(video_path, video_name, dims, startfrm):
    fp = open(os.path.join(video_path, video_name+'.yuv'), 'rb')
    blk_size = np.prod(dims) * 3 / 2
    fp.seek(np.int(blk_size * startfrm), 0)
    d00 = dims[0] // 2
    d01 = dims[1] // 2

    # Y channel
    y = fp.read(dims[0] * dims[1])
    y = np.frombuffer(y, dtype=np.uint8)
    y = y.reshape(dims)
    # U channel
    u = fp.read(d00 * d01)
    u = np.frombuffer(u, dtype=np.uint8)
    u = u.reshape(d00, d01)
    # V channel
    v = fp.read(d00 * d01)
    v = np.frombuffer(v, dtype=np.uint8)
    v = v.reshape(d00, d01)
    # scale UV channel
    enlarge_u = cv2.resize(u, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    enlarge_v = cv2.resize(v, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # to YUV channels to BGR channels
    img_yuv = cv2.merge([y, enlarge_u, enlarge_v])
    # img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) / 255.0
    img_rgb = YUV2RGB(img_yuv) / 255.0

    fp.close()
    return img_rgb


if __name__ == '__main__':
    # test imresize function
    # read images
    img = cv2.imread('test.png')
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # imresize
    scale = 1 / 4
    import time
    total_time = 0
    for i in range(10):
        start_time = time.time()
        rlt = imresize(img, scale, antialiasing=True)
        use_time = time.time() - start_time
        total_time += use_time
    print('average time: {}'.format(total_time / 10))

    import torchvision.utils
    torchvision.utils.save_image((rlt * 255).round() / 255, 'rlt.png', nrow=1, padding=0,
                                 normalize=False)
