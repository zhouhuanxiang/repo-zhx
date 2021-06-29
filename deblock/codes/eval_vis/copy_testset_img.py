import numpy as np
from numpy import loadtxt
import os

testlist = loadtxt('/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sep_testlist.txt', dtype=str)

print(testlist[:10])

for path in testlist:
    # src_path = os.path.join('/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences', path)
    # dst_path = os.path.join('/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_test', path)
    src_path = os.path.join('/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_blocky42', path)
    dst_path = os.path.join('/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_blocky42_test', path)
    os.makedirs(dst_path, exist_ok=True)
    os.system('cp -r {}/*.png {}'.format(src_path, dst_path))
    # print(dst_path)


# python ~/zhouhuanxiang/mmsr/codes/eval_vis/copy_testset_img.py



