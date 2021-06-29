"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'LQ':
        from data.LQ_dataset import LQDataset as D
    elif mode == 'LQVar':
        from data.LQ_var_dataset import LQVarDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    elif mode == 'LQGTDeblur':
        from data.LQGT_deblur_dataset import LQGTDataset as D
    elif mode == 'LQGTVmaf':
        from data.LQGT_vmaf_dataset import LQGTVmafDataset as D
    elif mode == 'LQGTVar':
        from data.LQGT_var_dataset import LQGTVarDataset as D
    # datasets for video restoration
    elif mode == 'REDS':
        from data.REDS_dataset import REDSDataset as D
    elif mode == 'Vimeo90K':
        from data.Vimeo90K_dataset import Vimeo90KDataset as D
    elif mode == 'video_test':
        from data.video_test_dataset import VideoTestDataset as D
    elif mode == 'KWAI':
        from data.KWAI_dataset import KWAIDataset as D
    elif mode == 'KWAI_Multi':
        from data.KWAI_multi_dataset import KWAIMultiDataset as D
    elif mode == 'Vimeo90K_test':
        from data.Vimeo90K_test_dataset import VimeoTestDataset as D
    elif mode == 'Vimeo90K_LQ':
        from data.Vimeo90K_LQ_dataset import VimeoLQDataset as D
    elif mode == 'Vimeo90K_LQ_test':
        from data.Vimeo90K_LQ_test_dataset import VimeoLQTestDataset as D
    elif mode == 'Xiph':
        from data.Xiph_data import XiphDataset as D
    elif mode == 'Xiph_Cutblur':
        from data.Xiph_Cutblur_data import XiphCutBlurDataset as D
    elif mode == 'Xiph3':
        from data.Xiph3_data import Xiph3Dataset as D
    elif mode == 'Xiph3Mix2':
        from data.Xiph3_mix2_data import Xiph3Mix2Dataset as D
    elif mode == 'Xiph_test':
        from data.Xiph_test_data import XiphTestDataset as D
    elif mode == 'Xiph3_test':
        from data.Xiph3_test_data import Xiph3TestDataset as D
    elif mode == 'Xiph_semi_A':
        from data.Xiph_semi_data_A import XiphSemiDatasetA as D
    elif mode == 'Xiph_semi_B':
        from data.Xiph_semi_data_B import XiphSemiDatasetB as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
