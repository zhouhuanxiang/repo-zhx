# [1]How to read metainfo from y4m?
# https://github.com/vigata/y4mtools

# [2]How to convert y4m to yuv?
# https://stackoverflow.com/a/16859790/8335078

# [3]How to use HM & HEVC?
# https://blog.csdn.net/lin453701006/article/details/52775820

# [4]Xiph website
# https://media.xiph.org/video/derf/

import os
import re
import sys
import glob
import pprint
import json

def get_y4m_metainfo(yuv60_path, y4m_path):

    y4m_metainfos = {}

    with open(yuv60_path) as fp:
        line = fp.readline()
        while line:
            # example: stefan_sif\t352\t240\t300\t1\n
            print(line.strip())
            words = line.strip().split(' ')
            print(words)
            y4m_name = words[0]
            width = words[1]
            height = words[2]
            length = words[3]

            y4m_metainfos[y4m_name] = {
                'height': height,
                'width': width,
                'length': length
            }

            line = fp.readline()
    return y4m_metainfos

def encode_yuv(y4m_path, y4m_metainfos, result_path, hm_path, qp=32):
    # write config file first
    fp = open('/home/web_server/zhouhuanxiang/disk/HM-16.0/cfg/encoder_lowdelay_P_main.cfg')
    lowdelay_cfg = fp.readlines()

    os.makedirs('/home/web_server/zhouhuanxiang/disk/Xiph/xiph_cfg_{}'.format(str(qp)), exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    print(y4m_metainfos)
    count = -1
    for y4m_name, metainfo in y4m_metainfos.items():

        count += 1
        if count % 2 != 0:
            continue

        # if y4m_name.find('Kimono1') >= 0:
        #     continue

        real_name = y4m_name
        width = metainfo['width']
        height = metainfo['height']

        print(real_name, width, height)
        print('***********\n')

        # skip 
        # if os.path.exists(os.path.join(result_path, real_name+'.yuv')) and not real_name in y4m_720p_list:
        #     continue
        # else:
        #     print('***********\n')

        header = '#======== File I/O ===============\n'\
            'InputFile: {}\n'\
            'InputBitDepth: 8\n'\
            'InputChromaFormat: 420\n'\
            'FrameRate: 30\n'\
            'FrameSkip: 0\n'\
            'SourceWidth: {}\n'\
            'SourceHeight: {}\n'\
            'FramesToBeEncoded: {}\n\n'\
            'BitstreamFile: {}\n'\
            'ReconFile: {}\n\n'.format(
                os.path.join(y4m_path, real_name+'.yuv'),
                width,
                height,
                metainfo['length'],
                os.path.join(result_path, real_name+'.bin'),
                os.path.join(result_path, real_name+'.yuv')
            )
        
        body = ''
        for i in range(4, 37):
            body += lowdelay_cfg[i]
        body += 'QP: {}\n'.format(qp)
        for i in range(38, 116):
            body += lowdelay_cfg[i]

        with open(os.path.join('/home/web_server/zhouhuanxiang/disk/Xiph','xiph_cfg_'+str(qp), real_name+'.cfg'), 'w') as fp:
            fp.write(header)
            fp.write(body)

        os.system('{} -c {}'.format(
            hm_path,
            os.path.join('/home/web_server/zhouhuanxiang/disk/Xiph', 'xiph_cfg_'+str(qp), real_name+'.cfg')
        ))


def main(argv):
    yuv60_path = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/YUV_test_kimono.txt'
    # yuv60_path = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/YUV_test_10.txt'
    y4m_path = argv[1]
    result_path = argv[2]
    hm_path = argv[3]

    y4m_metainfos = get_y4m_metainfo(yuv60_path, y4m_path)
    # with open('/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_test_metainfos.json', 'w') as fp:
    #     json.dump(y4m_metainfos, fp, indent=4)

    encode_yuv(y4m_path, y4m_metainfos, result_path, hm_path, qp=37)


if __name__ == '__main__':
    main(sys.argv)

'''Test
python /home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/encode_xiph_test_with_hm16.py \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_encoded \
/home/web_server/zhouhuanxiang/disk/HM-16.0/bin/TAppEncoderStatic
'''

'''37
nohup python /home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/encode_xiph_test_with_hm16.py \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded_37 \
/home/web_server/zhouhuanxiang/disk/HM-16.0/bin/TAppEncoderStatic > ~/zhouhuanxiang/encode37_kimono 2>&1 &
'''

'''42
nohup python /home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/encode_xiph_test_with_hm16.py \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_all_encoded42 \
/home/web_server/zhouhuanxiang/disk/HM-16.0/bin/TAppEncoderStatic > ~/zhouhuanxiang/encode42_kimono 2>&1 &
'''

'''Kimono
python /home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/encode_xiph_test_with_hm16.py \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_test_encoded \
/home/web_server/zhouhuanxiang/disk/HM-16.0/bin/TAppEncoderStatic
'''

