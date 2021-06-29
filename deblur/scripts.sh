CUDA_VISIBLE_DEVICES=1,2,3,4 python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai.yml
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai.yml > ~/zhouhuanxiang/disk/train_kwai_2 2>&1 &

CUDA_VISIBLE_DEVICES=2 python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_u2net_2.yml
CUDA_VISIBLE_DEVICES=2 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_u2net_2.yml > ~/zhouhuanxiang/disk/train_kwai_u2net_2 2>&1 &


CUDA_VISIBLE_DEVICES=1  python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_noise.yml
CUDA_VISIBLE_DEVICES=3  python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_unet_attn_noise_3.yml

CUDA_VISIBLE_DEVICES=2 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_u2net_3.yml > ~/zhouhuanxiang/disk/train_kwai_u2net_3 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_u2net_3_noise.yml > ~/zhouhuanxiang/disk/train_kwai_u2net_3_noise 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_unet_attn.yml > ~/zhouhuanxiang/disk/train_kwai_unet_attn 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_unet_attn_noise.yml > ~/zhouhuanxiang/disk/train_kwai_unet_attn_noise 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_unet_attn_noise_2.yml > ~/zhouhuanxiang/disk/train_kwai_unet_attn_noise_2 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_unet_attn_noise_3.yml > ~/zhouhuanxiang/disk/train_kwai_unet_attn_noise_3 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_unet_attn_noise_23.yml > ~/zhouhuanxiang/disk/train_kwai_unet_attn_noise_23 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python ~/zhouhuanxiang/disk/Real-SR-master/codes/train.py -opt \
~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/train_kwai_unet_attn_noise_34.yml > ~/zhouhuanxiang/disk/train_kwai_unet_attn_noise_34 2>&1 &

CUDA_VISIBLE_DEVICES=6 python train.py -opt ./options/kwai/train_u2net_x3_kwai.yml
CUDA_VISIBLE_DEVICES=6 nohup python train.py -opt ./options/kwai/train_u2net_x1_combined.yml > ~/zhouhuanxiang/disk/train_u2net_x1_combined 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py -opt ./options/kwai/train_u2net_x1_iphone.yml > ~/zhouhuanxiang/disk/train_u2net_x1_iphone 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py -opt ./options/kwai/train_u2net_x1_kwai.yml > ~/zhouhuanxiang/disk/train_u2net_x1_kwai 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python train.py -opt ./options/kwai/train_u2net_x3_combined.yml > ~/zhouhuanxiang/disk/train_u2net_x3_combined 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python train.py -opt ./options/kwai/train_u2net_x3_iphone.yml > ~/zhouhuanxiang/disk/train_u2net_x3_iphone 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python train.py -opt ./options/kwai/train_u2net_x3_kwai.yml > ~/zhouhuanxiang/disk/train_u2net_x3_kwai 2>&1 &



CUDA_VISIBLE_DEVICES=6 python test.py -opt ./options/kwai/test_u2net_3.yml
CUDA_VISIBLE_DEVICES=7 python test.py -opt ./options/kwai/archived/dped2kwai.yml
CUDA_VISIBLE_DEVICES=6 python test.py -opt ./options/kwai/test_mpr.yml



CUDA_VISIBLE_DEVICES=0 python ~/zhouhuanxiang/disk/Real-SR-master/codes/test.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/test_dped_1.yml
CUDA_VISIBLE_DEVICES=1 python ~/zhouhuanxiang/disk/Real-SR-master/codes/test.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/test_dped_2.yml
CUDA_VISIBLE_DEVICES=2 python ~/zhouhuanxiang/disk/Real-SR-master/codes/test.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/test_u2net_2.yml

CUDA_VISIBLE_DEVICES=5 python ~/zhouhuanxiang/disk/Real-SR-master/codes/test.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/test_u2net_3.yml
CUDA_VISIBLE_DEVICES=1 python ~/zhouhuanxiang/disk/Real-SR-master/codes/test.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/test_unet_attn.yml


CUDA_VISIBLE_DEVICES=5 python ~/zhouhuanxiang/disk/Real-SR-master/codes/test.py -opt ~/zhouhuanxiang/disk/Real-SR-master/codes/options/kwai/test_dpdd.yml




