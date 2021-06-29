import pickle
import glob
import os
import sys
import argparse
import socket
import cv2
import xml.dom.minidom as minidom
import subprocess
        

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
    # os.system('ffmpeg -r 30 -i {}/img_%5d.png -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 6 -preset fast -c:a copy -profile:a aac_he -ac 2 -x265-params lossless=1  -tag:v hvc1  -pix_fmt yuv420p -y {}'
    #     .format(imgs_path, video_path))
    os.system(
        'ffmpeg -r 30 -i {}/img_%5d.png -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 6 -preset fast -crf 10 -tag:v hvc1  -pix_fmt yuv420p -y {}'
        .format(imgs_path, video_path))

def imgs2videofps(imgs_path, video_path, fps):
    # fps fixed to 30
    # os.system('ffmpeg -r 30 -i {}/img_%5d.png -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 6 -preset fast -c:a copy -profile:a aac_he -ac 2 -x265-params lossless=1  -tag:v hvc1  -pix_fmt yuv420p -y {}'
    #     .format(imgs_path, video_path))
    os.system(
        '/home/web_server/zhouhuanxiang/disk/ffmpeg/bin/ffmpeg -r {} -i {}/%5d.png -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 6 -preset fast -crf 10 -tag:v hvc1  -pix_fmt yuv420p -y {}'
        .format(fps, imgs_path, video_path))

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

python ~/zhouhuanxiang/mmsr/codes/data_scripts/0_rawframes_tmp.py

nohup python ~/zhouhuanxiang/mmsr/codes/data_scripts/0_rawframes_tmp.py > ~/zhouhuanxiang/dump 2>&1 &


'''

def main1(args):
    img_paths = glob.glob('/home/web_server/zhouhuanxiang/disk/log/results/test_20210103/LQDeblur/hot_blur_qualified/*')
    for img_path in img_paths:
        name = img_path.split('/')[-1]
        video_folder = os.path.join('/home/web_server/zhouhuanxiang/disk/log/results/test_20210103/LQDeblur-video')
        os.makedirs(video_folder, exist_ok=True)

        source_video_path = os.path.join('/media/disk1/fordata/web_server/huangxiaozheng/hot_blur_qualified', name+'.mp4')
        source_video_fps = os.popen("ffmpeg -i " + source_video_path + " 2>&1 | sed -n \"s/.*, \(.*\) fp.*/\\1/p\"").read()
        print(source_video_fps)
        source_video_fps = round(float(source_video_fps))
        imgs2videofps(img_path, os.path.join(video_folder, name + '.mp4'), source_video_fps)
        print(os.path.join(video_folder, name + '.mp4'))

def main(args):

    # videos = glob.glob('/home/web_server/zhouhuanxiang/disk/data_test/testcase_gan_deart/output_medium/*.mp4')
    #
    # for video in videos:
    #     video_split = video.split('/')
    #     group = video_split[-2]
    #     name = video_split[-1].split('.')[0]
    #     # img_path = os.path.join('/mnt/video/maqiufang/tmp/testcase_gan_deart_raw', group, name)
    #     img_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data_test/testcase_gan_deart/', group+'_raw', name)
    #     os.makedirs(img_path, exist_ok=True)
    #     dumpFrames(video, name, img_path, False, False)

        # break
    img_paths = glob.glob('/home/web_server/zhouhuanxiang/disk/log/results/output_medium_20201112/KWAI-test/*')
    for img_path in img_paths:
        name = img_path.split('/')[-1]
        if len(name) != 11:
            continue
        group = img_path.split('/')[-3]
        video_folder = os.path.join('/home/web_server/zhouhuanxiang/disk/log/results', group, 'KWAI-test-video')
        # if group != 'test_SRGAN_phase2_patchls_o2m_var_KWAI35_2_output_medium_1':
        #     continue
        # continue
        os.makedirs(video_folder, exist_ok=True)

        source_group = 'output_medium'
        # source_group = 'output_vid'
        print(video_folder, source_group, name)
        source_video_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data_test/testcase_gan_deart', source_group, name+'.mp4')
        # print(source_video_path)
        source_video_fps = os.popen("ffmpeg -i " + source_video_path + " 2>&1 | sed -n \"s/.*, \(.*\) fp.*/\\1/p\"").read()
        print(source_video_fps)
        source_video_fps = round(float(source_video_fps))
        imgs2videofps(img_path, os.path.join(video_folder, name + '.mp4'), source_video_fps)
        print(os.path.join(video_folder, name + '.mp4'))
        # break



    # for video in videos:
    #     video_split = video.split('/')
    #     group = video_split[-2].split('_')[-1].split('.')[-1]
    #     name = video_split[-1].split('.')[0]
    #     # print(group, name)
    #     if not group in ['326', '48', '56']:
    #         continue
    #     img_path = os.path.join('/home/web_server/zhouhuanxiang/disk/log/results/test_SRGAN_phase2_patchls_o2m_var_KWAI35_2_0.'+group+'/KWAI-test', name)
    #     video_folder = os.path.join('/home/web_server/zhouhuanxiang/disk/log/results/test_SRGAN_phase2_patchls_o2m_var_KWAI35_2_0.'+group+'/KWAI-test-video')
    #     # video_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data_test/testcase_gan_deart', group, name+'.mp4')
    #     # print(img_path)
    #     # print(video_folder)
    #     os.makedirs(video_folder, exist_ok=True)
    #     imgs2video(img_path, os.path.join(video_folder, name + '.mp4'))


    # for dataset in ['HD_UGC_crf25_25', 'HD_UGC_crf25_30']:
    #     for vname in HD_UGC_train_list:
    #         video_path = os.path.join('/home/web_server/zhouhuanxiang/disk/data_test', dataset, vname)
    #         img_path   = os.path.join('/home/web_server/zhouhuanxiang/disk/data_test', dataset+'_raw', vname.split('.')[0])
    #         dumpFrames(video_path, vname, img_path, True, True)




if __name__ == '__main__':
    args = parse_args()
    main1(args)