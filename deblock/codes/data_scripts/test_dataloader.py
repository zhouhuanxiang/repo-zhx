import sys
import os.path as osp
import math
import torchvision.utils
import os

sys.path.insert(1, osp.dirname(osp.dirname(osp.abspath(__file__))))
from data import create_dataloader, create_dataset  # noqa: E402
from utils import util  # noqa: E402


def main():
    dataset = 'Xiph3Mix2'  # REDS | Vimeo90K | DIV2K800_sub
    opt = {}
    opt['dist'] = False
    opt['gpu_ids'] = [0]
    if dataset == 'LQGT':
        opt['name'] = 'test_KWAI'
        opt['dataroot_GT'] = '/home/web_server/zhouhuanxiang/disk/data/HD_UGC_raw'
        opt['dataroot_LQ'] = '/home/web_server/zhouhuanxiang/disk/data/HD_UGC_crf40_raw'
        opt['win_size'] = 11
        opt['win_type'] = 'gauss'
        opt['mode'] = 'LQGTVar'
        # opt['mode'] = 'LQGT'
        opt['color'] = 'RGB'
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 0
        opt['batch_size'] = 16
        opt['GT_size'] = 128
        opt['LQ_size'] = 128
        opt['scale'] = 1
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
        opt['data_type'] = 'img'  # img | lmdb | mc
    elif dataset == 'KWAI':
        opt['name'] = 'test_KWAI'
        # opt['dataroot_GT'] = '/home/web_server/zhouhuanxiang/disk/vdata/HD_UGC.lmdb'
        # opt['dataroot_LQ'] = '/home/web_server/zhouhuanxiang/disk/vdata/HD_UGC_crf40.lmdb'
        opt['dataroot_GT'] = '/home/web_server/zhouhuanxiang/disk/data/HD_UGC_raw'
        opt['dataroot_LQ'] = '/home/web_server/zhouhuanxiang/disk/data/HD_UGC_crf40_raw'
        opt['mode'] = 'KWAI_Multi'
        opt['N_frames'] = 5
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 1
        opt['batch_size'] = 4
        opt['GT_size'] = 256
        opt['LQ_size'] = 256
        opt['scale'] = 1
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
        opt['data_type'] = 'img'  # img | lmdb | mc
    elif dataset == 'REDS':
        opt['name'] = 'test_REDS'
        opt['dataroot_GT'] = '../../datasets/REDS/train_sharp_wval.lmdb'
        opt['dataroot_LQ'] = '../../datasets/REDS/train_sharp_bicubic_wval.lmdb'
        opt['mode'] = 'REDS'
        opt['N_frames'] = 5
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 8
        opt['batch_size'] = 16
        opt['GT_size'] = 256
        opt['LQ_size'] = 64
        opt['scale'] = 4
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
        opt['data_type'] = 'lmdb'  # img | lmdb | mc
    elif dataset == 'Vimeo90K':
        opt['name'] = 'test_Vimeo90K'
        opt['dataroot_GT'] = '../../datasets/vimeo90k/vimeo90k_train_GT.lmdb'
        opt['dataroot_LQ'] = '../../datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
        opt['mode'] = 'Vimeo90K'
        opt['N_frames'] = 7
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 8
        opt['batch_size'] = 16
        opt['GT_size'] = 256
        opt['LQ_size'] = 64
        opt['scale'] = 4
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
        opt['data_type'] = 'lmdb'  # img | lmdb | mc
    elif dataset == 'Vimeo90K_test':
        opt['name'] = 'vimeo90k-test'
        opt['dataroot_GT'] = '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences'
        opt['dataroot_LQ'] = '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_blocky32'
        opt['mode'] = 'Vimeo90K_test'
        opt['all_gt'] = True
        opt['N_frames'] = 7
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 8
        opt['batch_size'] = 16
        opt['GT_size'] = 256
        opt['LQ_size'] = 256
        opt['scale'] = 1
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
        opt['data_type'] = 'img'  # img | lmdb | mc
    elif dataset == 'Vimeo90K_LQ':
        opt['name'] = 'Vimeo90K-LQ'
        opt['dataroot_LHQ'] = '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_blocky32'
        opt['dataroot_LQ'] = '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_blocky37'
        opt['dataroot_LLQ'] = '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_blocky42'
        opt['mode'] = 'Vimeo90K_LQ'
        opt['patch_size'] = 32
        opt['patch_repeat'] = 5
        opt['N_frames'] = 7
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 8
        opt['batch_size'] = 16
        opt['GT_size'] = 256
        opt['LQ_size'] = 256
        opt['scale'] = 1
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
        opt['data_type'] = 'img'  # img | lmdb | mc
    elif dataset == 'Xiph3Mix2':
        opt['name'] = 'Xiph3Mix2'
        # opt['dataroot_LQ'] = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_37'
        # opt['dataroot_GT'] = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph'
        # opt['metainfo_path'] = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_metainfos_128.json'

        # opt['dataroot_LQ'] = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph_all_encoded_37'
        opt['dataroot_GT'] = '/home/web_server/zhouhuanxiang/disk/Xiph/Xiph'
        opt['metainfo_path'] = '/home/web_server/zhouhuanxiang/mmsr/codes/data_scripts/y4m_metainfos_128.json'
        opt['mode'] = 'Xiph3Mix2'
        opt['N_frames'] = 1
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 0
        opt['batch_size'] = 16
        opt['GT_size'] = 240
        opt['LQ_size'] = 240
        opt['scale'] = 1
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['interval_list'] = [1]
        opt['random_reverse'] = False
        opt['border_mode'] = False
        opt['cache_keys'] = None
    elif dataset == 'DIV2K800_sub':
        opt['name'] = 'DIV2K800'
        opt['dataroot_GT'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
        opt['dataroot_LQ'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4.lmdb'
        opt['mode'] = 'LQGT'
        opt['phase'] = 'train'
        opt['use_shuffle'] = True
        opt['n_workers'] = 8
        opt['batch_size'] = 16
        opt['GT_size'] = 128
        opt['scale'] = 4
        opt['use_flip'] = True
        opt['use_rot'] = True
        opt['color'] = 'RGB'
        opt['data_type'] = 'lmdb'  # img | lmdb
    else:
        raise ValueError('Please implement by yourself.')

    os.system('rm -rf /home/web_server/zhouhuanxiang/disk/test_tmp')
    os.makedirs('/home/web_server/zhouhuanxiang/disk/test_tmp')
    train_set = create_dataset(opt)
    train_loader = create_dataloader(train_set, opt, opt, None)
    nrow = int(math.sqrt(opt['batch_size']))
    padding = 2 if opt['phase'] == 'train' else 0

    util.set_random_seed(10)

    print('start...')
    # for i, data in enumerate(train_loader):
    #     if i > 5:
    #         break
    #     print(i)
    #
    #
    #     LQs = data['LQs']
    #     # LLQs = data['LLQs']
    #     # LHQs = data['LHQs']
    #     patch_labels = data['patch_labels']
    #     patch_offsets = data['patch_offsets']
    #     print(patch_labels.shape)
    #     print(patch_offsets.shape)
    #     print(LQs.shape)
    #
    #     for j in range(LQs.size(1)):
    #         torchvision.utils.save_image(LQs[:, j, :, :, :],
    #                                      '/home/web_server/zhouhuanxiang/disk/test_tmp/LQ_{:03d}_{}.png'.format(i, j), nrow=nrow,
    #                                      padding=padding, normalize=False)
    #     # for j in range(LLQs.size(1)):
    #     #     torchvision.utils.save_image(LLQs[:, j, :, :, :],
    #     #                                  '/home/web_server/zhouhuanxiang/disk/test_tmp/LLQ_{:03d}_{}.png'.format(i, j), nrow=nrow,
    #     #                                  padding=padding, normalize=False)
    #     # for j in range(LHQs.size(1)):
    #     #     torchvision.utils.save_image(LHQs[:, j, :, :, :],
    #     #                                  '/home/web_server/zhouhuanxiang/disk/test_tmp/LHQ_{:03d}_{}.png'.format(i, j), nrow=nrow,
    #     #                                  padding=padding, normalize=False)

    for i, data in enumerate(train_loader):
        if i > 2:
            break
        print(i)
        if dataset == 'REDS' or dataset == 'Vimeo90K' or dataset == 'KWAI' or dataset == 'Vimeo90K_test' or dataset == 'Xiph':
            LQs = data['LQs']
        else:
            LQ = data['LQ']
            # LQ = data['LQ_sigma']
            # print(LQ.shape)
            # LQ = LQ.repeat(1, 3, 1, 1)
        if dataset == 'Vimeo90K_test' and opt['all_gt']:
            GTs = data['GTs']
        else:
            GT = data['GT']
            # GT = data['LQ']

        if dataset == 'REDS' or dataset == 'Vimeo90K' or dataset == 'KWAI' or dataset == 'Vimeo90K_test' or dataset == 'Xiph':
            for j in range(LQs.size(1)):
                print(j)
                torchvision.utils.save_image(LQs[:, j, :, :, :],
                                             '/home/web_server/zhouhuanxiang/disk/test_tmp/{:03d}_{}_LQ.png'.format(i, j), nrow=nrow,
                                             padding=padding, normalize=False)
        else:
            torchvision.utils.save_image(LQ, '/home/web_server/zhouhuanxiang/disk/test_tmp/{:03d}_LQ.png'.format(i), nrow=nrow,
                                         padding=padding, normalize=False)

        if dataset == 'Vimeo90K_test' and opt['all_gt']:
            for j in range(GTs.size(1)):
                torchvision.utils.save_image(GTs[:, j, :, :, :],
                                             '/home/web_server/zhouhuanxiang/disk/test_tmp/{:03d}_{}_GT.png'.format(i, j),
                                             nrow=nrow,
                                             padding=padding, normalize=False)
        else:
            torchvision.utils.save_image(GT, '/home/web_server/zhouhuanxiang/disk/test_tmp/{:03d}_GT.png'.format(i), nrow=nrow, padding=padding,
                                         normalize=False)


if __name__ == "__main__":
    main()


# python ~/zhouhuanxiang/mmsr/codes/data_scripts/test_dataloader.py