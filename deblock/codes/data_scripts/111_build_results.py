import os
import glob
import socket
import argparse
import xml.dom.minidom as minidom
import numpy as np
from numpy import linalg
import pickle
from skimage import io, measure, color



test_video_ids = [
    '15398963809', '15407349694', '15439392361', '15488420682', '15523784704', '16121597861', '16064993671', '15994056528',
    '15406724531', '15627873950', '15690684867', '15752323646', '15991733137', '15898181711', '15836862878'
]


def parse_args():
    parser = argparse.ArgumentParser(description='extract raw frames')
    parser.add_argument('--models', nargs='+', 
                        help='tested models')
    parser.add_argument('--crfs', nargs='+', 
                        help='crfs of tested datasets')
    args = parser.parse_args()
    return args


def imgs2video(imgs_path, video_path):
    # fps fixed to 30
    os.system('ffmpeg -r 30 -i {}/img_%5d.png -movflags +faststart -max_interleave_delta 150000000 -max_muxing_queue_size 9999 -c:v libx265 -psnr -threads 6 -preset fast -c:a copy -profile:a aac_he -ac 2 -x265-params lossless=1  -tag:v hvc1  -pix_fmt yuv420p -y {}'
        .format(imgs_path, video_path))


def concatVideos(left_video_path, right_video_path, result_video_path):
    os.system('ffmpeg -i {} -i {} -filter_complex "[0:v:0]pad=iw*2:ih[bg]; [bg][1:v:0]overlay=w" -qp 10 -y {}'
        .format(left_video_path, right_video_path, result_video_path))


def getVmaf(model, vid, video_path):
    xml_path = '/home/web_server/zhouhuanxiang/disk/log/results/{}/KWAI-test-xml'.format(model)
    os.makedirs(xml_path, exist_ok=True)
    # os.system('ffmpeg -i {} -i {} -lavfi libvmaf="model_path=/home/web_server/zhouhuanxiang/disk/vmaf/model/vmaf_v0.6.1.pkl:log_path={}:psnr=true:ssim=true" -hide_banner -f null -'
    #     .format(
    #         video_path,
    #         os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_align', vid+'.mp4'),
    #         os.path.join(xml_path, vid+'.xml')
    #     ))
    #
    # xml_file = minidom.parse(os.path.join(xml_path, vid+'.xml'))
    # xml_item = xml_file.getElementsByTagName('fyi')[0]
    #
    # vmaf = float(xml_item.attributes['aggregateVMAF'].value)
    # psnr = float(xml_item.attributes['aggregatePSNR'].value)
    # ssim = float(xml_item.attributes['aggregateSSIM'].value)

    os.system(
        'ffmpeg -i {} -i {} -lavfi libvmaf="model_path=/home/web_server/zhouhuanxiang/disk/vmaf/model/vmaf_v0.6.1.pkl:log_path={}" -hide_banner -f null -'
            .format(
            video_path,
            os.path.join('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_align', vid + '.mp4'),
            os.path.join(xml_path, vid + '.xml')
        ))

    xml_file = minidom.parse(os.path.join(xml_path, vid + '.xml'))
    xml_item = xml_file.getElementsByTagName('fyi')[0]

    vmaf = float(xml_item.attributes['aggregateVMAF'].value)

    return vmaf


def getPSNR_SSIM(model):
    img_dir = '/home/web_server/zhouhuanxiang/disk/log/results/{}/KWAI-test'.format(model)
    img1_paths = glob.glob(os.path.join(img_dir, '*', '*.png'))
    img0_paths = [i.replace('log/results/{}/KWAI-test'.format(model), 'data/HD_UGC_raw') for i in img1_paths]

    img0_paths.sort()
    img1_paths.sort()

    ssims = []
    psnrs = []
    for img0_path, img1_path in list(zip(img0_paths, img1_paths)):
        if img0_path.find('15398963809') == -1:
            continue

        print(img0_path, '\n', img1_path, '\n\n')
        img0 = io.imread(img0_path)
        img1 = io.imread(img1_path)

        img0 = color.rgb2ycbcr(img0)[:, :, 0]
        img1 = color.rgb2ycbcr(img1)[:, :, 0]
        # color.ycbcr2rgb()
        # color.yuv2rgb()

        ycbcr_from_rgb = np.array([[65.481, 128.553, 24.966],
                                   [-37.797, -74.203, 112.0],
                                   [112.0, -93.786, -18.214]])
        ycbcr_from_rgb = ycbcr_from_rgb / 255.0
        rgb_from_ycbcr = linalg.inv(ycbcr_from_rgb)

        print(rgb_from_ycbcr)
        break

        # print(img0)

        ssim = measure.compare_ssim(img0, img1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=False)
        psnr = measure.compare_psnr(img0, img1, data_range=220)
        ssims.append(ssim)
        psnrs.append(psnr)

        print(ssim, psnr)

    return np.mean(psnrs), np.mean(ssims)


'''
python ~/zhouhuanxiang/mmsr/codes/data_scripts/111_build_results.py \
--models test_SRResNet_KWAI35_baseline

python ~/zhouhuanxiang/mmsr/codes/data_scripts/111_build_results.py \
--models test_SRGAN_phase2_patchls_KWAI253035_2

python ~/zhouhuanxiang/mmsr/codes/data_scripts/111_build_results.py \
--models test_EDSR_KWAI40_Xiph42 test_SRResNet_var_gauss_KWAI40 test_SRResNet_var_mean_KWAI40 \

python ~/zhouhuanxiang/mmsr/codes/data_scripts/111_build_results.py \
--models test_SRGAN_phase2_patchls_KWAI35_2 test_SRGAN_phase2_patchls_KWAI35_2_50k \

python ~/zhouhuanxiang/mmsr/codes/data_scripts/111_build_results.py \
--models test_SRGAN_phase2_patchls_o2m_KWAI35_2 test_SRGAN_phase2_patchls_o2m_KWAI35_2 \

python ~/zhouhuanxiang/mmsr/codes/data_scripts/111_build_results.py \
--models test_SRGAN_phase2_patchls_KWAI35_2_190k test_SRGAN_phase2_patchls_perceptual_KWAI35_1_75000 test_SRGAN_phase2_patchls_perceptual_KWAI35_2_75000 \

python ~/zhouhuanxiang/mmsr/codes/data_scripts/111_build_results.py --models test_SRGAN_phase2_patchls_o2m_KWAI35_2_0.01

python ~/zhouhuanxiang/mmsr/codes/data_scripts/111_build_results.py --models test_SRGAN_phase2_patchls_o2m_spectralGD_KWAI35_2 
'''

def main(args):
    models = args.models
    crfs = args.crfs
    for i, model in enumerate(models):
        img_path = '/home/web_server/zhouhuanxiang/disk/log/results/{}/KWAI-test'.format(model)
        video_path = '/home/web_server/zhouhuanxiang/disk/log/results/{}/KWAI-test-video'.format(model)
        os.makedirs(video_path, exist_ok=True)
        if crfs:
            compare_path = '/home/web_server/zhouhuanxiang/disk/log/results/test_SRResNet_KWAI{}_baseline/KWAI-test-video'.format(crfs[i])
            concat_video_path = '/home/web_server/zhouhuanxiang/disk/log/results/{}/KWAI-test-video-concat'.format(model)
            os.makedirs(concat_video_path, exist_ok=True)
        else:
            compare_path = ''
            concat_video_path = ''
        
        with open('/home/web_server/zhouhuanxiang/disk/data/HD_UGC_test_list', 'rb') as fp:
            vids = pickle.load(fp)
        vids = [vid.split('.')[0] for vid in vids] 
        print(vids)

        vmafs = []

        for vid in vids:
            # if vid != '15836862878':
            #     continue
        
            imgs2video(os.path.join(img_path, vid), os.path.join(video_path, vid+'.mp4'))
        
            # if compare_path:
            #     concatVideos(os.path.join(video_path, vid+'.mp4'),
            #         os.path.join(compare_path, vid+'.mp4'),
            #         os.path.join(concat_video_path, vid+'.mp4'))
        
            # continue

            # vmaf = getVmaf(model, vid, os.path.join(video_path, vid+'.mp4'))
            # vmafs.append(vmaf)
            # print(' #vmaf: {}\n'.format(vmaf))

        vmaf = np.mean(vmafs)
        print(' #vmaf mean: {}\n'.format(vmaf))
        # psnr, ssim = getPSNR_SSIM(model)
        # print(' #psnr mean: {}\n #ssim mean: {}\n'.format(psnr, ssim))



if __name__ == '__main__':
    args = parse_args()
    main(args)

    print('done')