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
    
y4m_422p_list = [
    'controlled_burn_1080p', 
    'football_422_cif', 
    'touchdown_pass_1080p', 
    'rush_field_cuts_1080p'
]

y4m_720p_list = [
    '720p50_shields_ter',
    '720p5994_stockholm_ter'
]

y4m_ntsc_list = [
    'mobile_calendar_422_4sif',
    'football_422_4sif'
]


def get_y4m_metainfo(yuv60_path, y4m_path):

    y4m_metainfos = {}

    with open(yuv60_path) as fp:
        line = fp.readline()
        while line:
            # example: stefan_sif\t352\t240\t300\t1\n
            print(line.strip())
            words = line.strip().split('\t')
            y4m_name = words[0].replace('2048x1080_60fps', '4096x2160_60fps_10bit')
            width = words[1]
            height = words[2]
            length = words[3]

            # check width and height directly from y4m metainfo
            y4m_file = open(os.path.join(y4m_path, y4m_name+'.y4m'), 'rb')
            header = y4m_file.readline()
            header = header.decode("utf-8")
            print('metainfo', header.strip())
            width1 = (re.compile("W(\d+)").findall(header))[0]
            height1 = (re.compile("H(\d+)").findall(header))[0]
            if height != height1 or width != width1:
                print('##### warnning #####\n')

            y4m_metainfos[y4m_name] = {
                'height': height1,
                'width': width1,
                'length': length
            }

            line = fp.readline()
    return y4m_metainfos


def convert_y4m2yuv(y4m_path, y4m_metainfos):
    for y4m_name in y4m_metainfos.keys():
        if y4m_name in y4m_ntsc_list:
            y4m_420p_name = y4m_name + '_420p'
            # 422p -> 420p
            os.system('ffmpeg -i {}.y4m -vf format=yuv420p -y {}.y4m'.format(
                os.path.join(y4m_path, y4m_name),
                os.path.join(y4m_path, y4m_420p_name)
            ))
            # y4m -> yuv
            os.system('ffmpeg -i {}.y4m {}.yuv -y'.format(
                os.path.join(y4m_path, y4m_420p_name),
                os.path.join(y4m_path, y4m_name)
            ))
            # rm y4m_420p
            os.system('rm {}.y4m'.format(os.path.join(y4m_path, y4m_420p_name)))
        # elif y4m_name in y4m_422p_list or y4m_name in y4m_720p_list:
        #     y4m_420p_name = y4m_name + '_420p'
        #     # 422p -> 420p
        #     os.system('ffmpeg -i {}.y4m -vf format=yuv420p -y {}.y4m'.format(
        #         os.path.join(y4m_path, y4m_name),
        #         os.path.join(y4m_path, y4m_420p_name)
        #     ))
        #     # y4m -> yuv
        #     os.system('ffmpeg -i {}.y4m {}.yuv -y'.format(
        #         os.path.join(y4m_path, y4m_420p_name),
        #         os.path.join(y4m_path, y4m_name)
        #     ))
        #     # rm y4m_420p
        #     os.system('rm {}.y4m'.format(os.path.join(y4m_path, y4m_420p_name)))
        # elif y4m_name.find('4096x2160_60fps_10bit') > 0:
        #     y4m_8bit_4k_name = y4m_name.replace('10bit', '8bit')
        #     y4m_8bit_2k_name = y4m_8bit_4k_name.replace('4096x2160', '2048x1080')
        #     yuv_8bit_2k_name = y4m_8bit_2k_name
        #     print(y4m_8bit_4k_name+'\n'+y4m_8bit_2k_name+'\n'+yuv_8bit_2k_name+'\n\n')
        #     # 10bit -> 8bit
        #     os.system('ffmpeg -i {}.y4m -vf format=yuv420p -y {}.y4m'.format(
        #         os.path.join(y4m_path, y4m_name),
        #         os.path.join(y4m_path, y4m_8bit_4k_name)
        #     ))
        #     # 4k -> 2k
        #     os.system('ffmpeg -i {}.y4m -vf scale=2048:1080 -y {}.y4m'.format(
        #         os.path.join(y4m_path, y4m_8bit_4k_name),
        #         os.path.join(y4m_path, y4m_8bit_2k_name)
        #     ))
        #     # y4m -> yuv
        #     os.system('ffmpeg -i {}.y4m -y {}.yuv'.format(
        #         os.path.join(y4m_path, y4m_8bit_2k_name),
        #         os.path.join(y4m_path, yuv_8bit_2k_name)
        #     ))
        #     # remove y4m_8bit
        #     os.system('rm {}.y4m'.format(os.path.join(y4m_path, y4m_8bit_4k_name)))
        #     # remove y4m_8bit_2k
        #     os.system('rm {}.y4m'.format(os.path.join(y4m_path, y4m_8bit_2k_name)))
        # else:
        #     # y4m -> yuv
        #     os.system('ffmpeg -i {}.y4m {}.yuv -y'.format(
        #         os.path.join(y4m_path, y4m_name),
        #         os.path.join(y4m_path, y4m_name)
        #     ))
        

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
        if count % 6 != 5:
            continue

        if y4m_name.find('4096x2160_60fps_10bit') > 0:
            real_name = y4m_name.replace('4096x2160_60fps_10bit', '2048x1080_60fps_8bit')
            width = str(int(metainfo['width']) / 2)
            height = str(int(metainfo['height']) / 2)
        else: 
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


def encode_yuv_parallel():
    cfg_files = glob.glob('')

def main(argv):
    yuv60_path = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/YUV_60.txt'
    y4m_path = argv[1]
    result_path = argv[2]
    hm_path = argv[3]

    y4m_metainfos = get_y4m_metainfo(yuv60_path, y4m_path)
    with open('/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_metainfos.json', 'w') as fp:
        json.dump(y4m_metainfos, fp, indent=4)

    # convert_y4m2yuv(y4m_path, y4m_metainfos)

    encode_yuv(y4m_path, y4m_metainfos, result_path, hm_path, qp=37)


if __name__ == '__main__':
    main(sys.argv)

'''
python /home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/encode_xiph_with_hm16.py \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_encoded \
/home/web_server/zhouhuanxiang/disk/HM-16.0/bin/TAppEncoderStatic
'''

'''
nohup python /home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/encode_xiph_with_hm16.py \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded \
/home/web_server/zhouhuanxiang/disk/HM-16.0/bin/TAppEncoderStatic > ~/zhouhuanxiang/encode 2>&1 &
'''

'''
nohup python /home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/encode_xiph_with_hm16.py \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph \
/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_37 \
/home/web_server/zhouhuanxiang/disk/HM-16.0/bin/TAppEncoderStatic > ~/zhouhuanxiang/encode5 2>&1 &
'''
