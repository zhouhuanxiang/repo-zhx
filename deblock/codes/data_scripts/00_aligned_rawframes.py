import pickle
import glob
import os
import sys
import argparse
import socket
import cv2

def dump_frames(full_path, vname, out_full_path, is_train=True):
    if os.path.exists(out_full_path):
        os.system('rm -rf '+out_full_path)
    os.makedirs(out_full_path)
    # Read frame by ffmpeg
    os.system('ffmpeg -i '+full_path+' '+out_full_path+'/img_%05d.png -hide_banner -vsync 0')
    # clean
    if is_train:
        imgs_path = glob.glob(os.path.join(out_full_path, '*.png'))
        imgs_path.sort()
        
        print('total_frames', len(imgs_path))
        img_array = cv2.imread(imgs_path[0])
        if len(imgs_path) < 150 or img_array.shape != (1280, 720, 3):
            imgs_path_1 = []
        else:
            border = (len(imgs_path) - 100) // 2
            imgs_path_1 = imgs_path[border:border+100]
        imgs_path_2 = list(set(imgs_path) - set(imgs_path_1))
        for img in imgs_path_2:
            os.system('rm {}'.format(img))
        for i, img1 in enumerate(imgs_path_1):
            # os.system('mv {} img_{:0>5d}.png'.format(img, i))
            i0 = 'img_{:0>5d}.png'.format(i)
            i1 = 'img_{:0>5d}.png'.format(i + border + 1)
            img0 = img1.replace(i1, i0)
            os.system('mv {} {}'.format(img1, img0))

    print('video {} extracted'.format(vname))
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract raw frames')
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--path', nargs='+',
                        help='path(s) of video folder to be extracted')
    parser.add_argument("--ext", type=str, default='mp4', choices=['avi', 'mp4', 'webm', 'flv'],
                        help='video file extensions')
    args = parser.parse_args()
    return args


'''
python ~/zhouhuanxiang/mmsr/codes/data_scripts/00_aligned_rawframes.py --path HD_UGC
python ~/zhouhuanxiang/mmsr/codes/data_scripts/00_aligned_rawframes.py --path HD_UGC_crf40 HD_UGC_crf25 
python ~/zhouhuanxiang/mmsr/codes/data_scripts/00_aligned_rawframes.py --path HD_UGC_crf30 HD_UGC_crf35 HD_UGC_crf45
python ~/zhouhuanxiang/mmsr/codes/data_scripts/00_aligned_rawframes.py --path HD_UGC_crf25_25 HD_UGC_crf25_30
'''

if __name__ == '__main__':
    args = parse_args()

    prefix = '/home/web_server/zhouhuanxiang/disk'

    with open (os.path.join(prefix, 'data', 'HD_UGC_list'), 'rb') as fp:
        HD_UGC_train_list = pickle.load(fp)
    with open (os.path.join(prefix, 'data', 'HD_UGC_test_list'), 'rb') as fp:
        HD_UGC_test_list = pickle.load(fp)
    with open (os.path.join(prefix, 'data', 'HD_UGC_deleted_list'), 'rb') as fp:
        HD_UGC_deleted_list = pickle.load(fp)

    for path in args.path:
        for vname in HD_UGC_train_list:
            full_path = os.path.join(prefix, 'vdata', path, vname)
            out_full_path = os.path.join(prefix, 'vdata', path+'_raw', vname.split('.')[0])
            dump_frames(full_path, vname, out_full_path, True)

        # for vname in HD_UGC_test_list:
        #     full_path = os.path.join(prefix, 'data', path, vname)
        #     out_full_path = os.path.join(prefix, 'data', path+'_raw_test', vname.split('.')[0])
        #     dump_frames(full_path, vname, out_full_path, False)

        #     os.makedirs(os.path.join(prefix, 'data', path+'_raw_test_aligned'), exist_ok=True)
        #     os.system('ffmpeg -r 30 -i {}/img_%5d.png -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 6 -preset fast -c:a copy -profile:a aac_he -ac 2 -x265-params lossless=1  -tag:v hvc1  -pix_fmt yuv420p -y {}'
        #                     .format(out_full_path, os.path.join(prefix, 'data', path+'_raw_test_aligned', vname)))