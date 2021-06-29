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

def YUV2RGB(yuv):
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
    img_rgb = YUV2RGB(img_yuv)

    fp.close()
    return img_rgb

roots = [
    # '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph',
    # '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_37',

    '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test',
    '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded_37'
]

# with open('/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_metainfos_128.json') as json_file:
with open('/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos.json') as json_file:
    metainfo_dict = json.load(json_file)
    for key in metainfo_dict.keys():
        metainfo_dict[key]['height'] = int(metainfo_dict[key]['height'])
        metainfo_dict[key]['width'] = int(metainfo_dict[key]['width'])
        metainfo_dict[key]['length'] = int(metainfo_dict[key]['length'])
    metainfos = dict(sorted(metainfo_dict.items(), key=lambda item: item[0]))

for root in roots:
    for key in metainfo_dict.keys():
        dst_root = root + '_img'
        os.makedirs(os.path.join(dst_root, key), exist_ok=True)

        video_height = metainfos[key]['height']
        video_width = metainfos[key]['width']
        video_length = metainfos[key]['length']
        for idx in range(video_length):
            img = read_yuv_frame(root, key, (video_height, video_width), idx)
            cv2.imwrite(os.path.join(dst_root, key, '{:08d}.png'.format(idx)), img)
            print(os.path.join(dst_root, key, '{:08d}.png'.format(idx)))
            # break

'''
python ~/zhouhuanxiang/mmsr/codes/data_scripts/5_extract_Xiph_image.py
'''