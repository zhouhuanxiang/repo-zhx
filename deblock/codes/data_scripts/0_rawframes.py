import pickle
import glob
import os
import sys
import argparse
import socket
import cv2
import xml.dom.minidom as minidom

test_video_ids = [
    '15398963809', '15407349694', '15439392361', '15488420682', '15523784704', '16121597861', '16064993671', '15994056528',
    '15406724531', '15627873950', '15690684867', '15752323646', '15991733137', '15898181711', '15836862878'
]

def alignTestVideo():
    with open('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_list', 'rb') as fp:
        HD_UGC_train_list = pickle.load(fp)
    with open('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_list', 'rb') as fp:
        HD_UGC_test_list = pickle.load(fp)

    os.makedirs('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_raw', exist_ok=True)
    os.makedirs('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_align', exist_ok=True)
    for vid in HD_UGC_train_list:
        # if vid != '15836862878.mp4':
        #     continue
        img_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_raw', vid.split('.')[0])
        if os.path.exists(img_path):
            os.system('rm -rf {}'.format(img_path))
        os.mkdir(img_path)
        os.system('ffmpeg -i {} {}/img_%05d.png -hide_banner'.format(
            os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC', vid),
            os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_raw', vid.split('.')[0])
        ))
    for vid in HD_UGC_train_list:
        # if vid != '15836862878.mp4':
        #     continue
        os.system('ffmpeg -r 30 -i {}/img_%5d.png -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 6 -preset fast -c:a copy -profile:a aac_he -ac 2 -x265-params lossless=1  -tag:v hvc1  -pix_fmt yuv420p -y {}'.format(
            os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_raw', vid.split('.')[0]),
            os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_align', vid)
        ))


def getVmaf(vid, video_path):
    os.system(
        'ffmpeg -i {} -i {} -lavfi libvmaf="model_path=/home/web_server/zhouhuanxiang/disk/vmaf/model/vmaf_v0.6.1.pkl:log_path={}" -hide_banner -f null -'
            .format(
            video_path,
            os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_align', vid),
            '/home/web_server/zhouhuanxiang/1.xml'
        ))
    xml_file = minidom.parse('/home/web_server/zhouhuanxiang/1.xml')
    xml_item = xml_file.getElementsByTagName('fyi')[0]
    vmaf = float(xml_item.attributes['aggregateVMAF'].value)
    return vmaf


def valideTrainData():
    with open('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_list', 'rb') as fp:
        HD_UGC_train_list = pickle.load(fp)
    with open('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_list', 'rb') as fp:
        HD_UGC_test_list = pickle.load(fp)
    
    vmafs = []
    for crf in ['25']:
        for vid in HD_UGC_train_list + HD_UGC_test_list:
            video_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_crf'+crf, vid)
            vmaf = getVmaf(vid, video_path)
            vmafs.append(vmaf)
    for vid, vmaf in list(zip(HD_UGC_train_list + HD_UGC_test_list, vmaf)):
        print(vid, '   ', vmaf)
        

def dumpFrames(video_path, vname, img_path, is_train=True, is_skip=True):
    if os.path.exists(img_path):
        os.system('rm -rf {}'.format(img_path))
    os.makedirs(img_path)
    # Read frame by ffmpeg
    os.system('ffmpeg -i {} {}/img_%05d.png -hide_banner'.format(video_path, img_path))
    # clean
    if is_train:
        if is_skip:
            imgs_path = glob.glob(os.path.join(img_path, '*.png'))
            imgs_path.sort()
            if len(img_path) < 50:
                img_path_1 = []
            else:
                imgs_path_1 = imgs_path[20:-20:5]
            imgs_path_2 = list(set(imgs_path) - set(imgs_path_1))
            for i in imgs_path_2:
                os.system('rm ' + i)
        else:
            imgs_path = glob.glob(os.path.join(img_path, '*.png'))
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


def compressVideo(src_video, crf, dst_video):
    os.system("ffmpeg -i {} -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 16 -preset fast -c:a copy -profile:a aac_he -ac 2 -crf {} -x265-params psnr=1:ssim=1:fakeparams=no  -tag:v hvc1  -pix_fmt yuv420p -y {}"
        .format(src_video, crf, dst_video))

def imgs2video(imgs_path, video_path):
    # fps fixed to 30
    os.system('ffmpeg -r 30 -i {}/img_%5d.png -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 6 -preset fast -c:a copy -profile:a aac_he -ac 2 -x265-params lossless=1  -tag:v hvc1  -pix_fmt yuv420p -y {}'
        .format(imgs_path, video_path))

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

python ~/zhouhuanxiang/mmsr/codes/data_scripts/0_rawframes.py

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/0_rawframes.py > ~/zhouhuanxiang/dump 2>&1 &


'''

def main(args):
    with open('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_list', 'rb') as fp:
        HD_UGC_train_list = pickle.load(fp)
    with open('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_list', 'rb') as fp:
        HD_UGC_test_list = pickle.load(fp)
    with open('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_deleted_list', 'rb') as fp:
        HD_UGC_deleted_list = pickle.load(fp)

    # alignTestVideo()
    # valideTrainData()
    # return

    # for crf in ['25']:
    #     os.makedirs('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_crf{}'.format(crf), exist_ok=True)
    #     for vname in HD_UGC_test_list:
    #         # if vname != '15836862878.mp4':
    #         #     continue
    #         src_video = os.path.join('/home/web_server/zhouhuanxiang/disk/data', 'HD_UGC', vname)
    #         dst_video = os.path.join('/home/web_server/zhouhuanxiang/disk/data', 'HD_UGC_crf'+crf, vname)
    #         compressVideo(src_video, crf, dst_video)

    # for dataset in ['HD_UGC_crf40', 'HD_UGC']:
    # for dataset in ['HD_UGC_crf25', 'HD_UGC_crf30', 'HD_UGC_crf35', 'HD_UGC_crf40', 'HD_UGC_crf45', 'HD_UGC']:
    # for dataset in ['HD_UGC_crf25', 'HD_UGC_crf30', 'HD_UGC_crf35', 'HD_UGC_crf45']:
    for dataset in ['HD_UGC_crf25_25', 'HD_UGC_crf25_30']:
        for vname in HD_UGC_train_list:
            video_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data', dataset, vname)
            img_path   = os.path.join('/home/web_server/zhouhuanxiang/disk/data', dataset+'_raw', vname.split('.')[0])
            dumpFrames(video_path, vname, img_path, True, True)

        # for vname in HD_UGC_test_list:
        #     # if vname != '15836862878.mp4':
        #     #     continue
        #     video_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data', dataset, vname)
        #     img_path   = os.path.join('/home/web_server/zhouhuanxiang/disk/data', dataset+'_raw_test', vname.split('.')[0])
        #     dumpFrames(video_path, vname, img_path, False)


# delete
    # for dataset in ['HD_UGC']:
    #     os.makedirs('/home/web_server/zhouhuanxiang/disk/data/{}_raw_test_aligned'.format(dataset), exist_ok=True)
    #     for vname in HD_UGC_test_list:
    #         # if vname != '15836862878.mp4':
    #         #     continue
    #         img_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data', dataset+'_raw_test', vname.split('.')[0])
    #         video_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data', dataset+'_raw_test_aligned', vname)
    #         imgs2video(img_path, video_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)