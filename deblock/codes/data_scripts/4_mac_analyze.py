import os
import os.path as osp
import sys
# import argparse
# import logging
# import json
# from collections import OrderedDict
# import pandas as pd
import cv2
import numpy as np
# import torch
# import torch.nn.functional as F

sys.path.insert(1, osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util  # noqa: E402
import utils.util



LQ_path = ''
HQ_path = ''
re_path = ''

for i in range(21):
    hq = cv2.imread('/Users/zhx/Desktop/Xiph_img/RaceHorses_416x240_30/{:05d}_HQ.png'.format(i)).astype('float')
    lq = cv2.imread('/Users/zhx/Desktop/Xiph_img/RaceHorses_416x240_30/{:05d}_LQ.png'.format(i)).astype('float')
    re = cv2.imread('/Users/zhx/Desktop/results/RaceHorses_416x240_30/{:05d}.png'.format(i)).astype('float')

    os.makedirs('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30', exist_ok=True)

    cv2.imwrite('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30/{:05d}_re.png'.format(i),
                np.abs(re - hq) * 3)
    cv2.imwrite('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30/{:05d}_lq.png'.format(i),
                np.abs(lq - hq))

    # cv2.imwrite('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30/{:05d}_r_re.png'.format(i), np.abs(re - hq)[:,:,0])
    # cv2.imwrite('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30/{:05d}_r_lq.png'.format(i), np.abs(lq - hq)[:,:,0])

    # cv2.imwrite('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30/{:05d}_g_re.png'.format(i), np.abs(re - hq)[:,:,1])
    # cv2.imwrite('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30/{:05d}_g_lq.png'.format(i), np.abs(lq - hq)[:,:,1])

    # cv2.imwrite('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30/{:05d}_b_re.png'.format(i), np.abs(re - hq)[:,:,2])
    # cv2.imwrite('/Users/zhx/Desktop/Xiph_compare/RaceHorses_416x240_30/{:05d}_b_lq.png'.format(i), np.abs(lq - hq)[:,:,2])





