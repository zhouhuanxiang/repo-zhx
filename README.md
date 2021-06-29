# repo-zhx

2018.9-2021.7 期间的工作总结

## 去压缩失真
#### 去压缩失真主要采用快手线上数据进行训练：
  - 采用ffmpeg进行压缩，人工构建压缩数据集；
  - 文件目录在29服务器上，/home/web_server/zhouhuanxiang/disk/data，数据集的规模为454，并人工选取其中有代表性的15个视频作为测试集；
  - 每个子文件夹代表特定压缩程度的视频集合，例如HD_UGC_crf30文件夹代表采用crf30进行压缩，可以自行选择合适的压缩程度。
#### 训练代码采用PyTorch作为基本框架：
  - 文件目录在29服务器上，/home/web_server/zhouhuanxiang/disk/mmsr；
  - 采用mmsr作为基本框架，mmsr目前已经被合并到mmediting框架中，具体可参考https://github.com/open-mmlab/mmediting；
  - 在实现上，没有对框架的主要流程进行修改，可直接参考原mmsr的训练、测试方式进行调用；
#### 训练结果：
  - 目前上线的模型为两阶段去压缩模糊模型，即在第一阶段去除块效应，第二节采用GAN增添细节信息；
  - 模型路径为，/home/web_server/zhouhuanxiang/disk/log/experiments
  - 测试结果路径为，/home/web_server/zhouhuanxiang/disk/log/results
#### 一些尝试：
  - vmaf损失函数	
    - 采用预训练模型模糊vmaf分数，利用该预训练模型作为额外的损失函数
    - ./models/models_sub/SR_vmaf_model.py
    - /home/web_server/zhouhuanxiang/disk/log/experiments/train_SRResNet_vmaf_G_KWAI35
  - SRResNet
    - 基本的残差网络模型 https://arxiv.org/abs/1707.02921
    - ./modes/archs/SRResNet.py
    - /home/web_server/zhouhuanxiang/disk/log/experiments/*SRResNet*
  - SRResNet_o2m
    - 基本的残差网络模型 + one-to-many 训练方式 https://arxiv.org/abs/1611.04994
    - ./models/archs/SRResNet_o2m.py
    - /home/web_server/zhouhuanxiang/disk/log/experiments/SRGANx1_phase2_patchls_o2m_KWAI35_2
  - SRResNet_o2m_var
    - 基本的残差网络模型 + one-to-many 训练方式 + 基于输入图像 variance 的噪声图 https://arxiv.org/abs/1611.04994
    - ./modes/archs/SRResNet_o2m_var.py
    - /home/web_server/zhouhuanxiang/disk/log/experiments/SRGANx1_phase2_patchls_o2m_var_KWAI35_0
  - EDVR
    - 利用时序信息，采用多帧模型EDVR https://arxiv.org/abs/1905.02716
    - ./models/archs/EDVR_arch.py
    - /home/web_server/zhouhuanxiang/disk/log/experiments/EDVR*

## 去模糊

#### 去模糊实验主要采用快手线上数据进行训练：
- 估计出来的模糊核： ~/huangxiaozheng/blur_kernel
- 生成模糊视频的matlab代码： ~/huangxiaozheng/blur_matlab
- 去模糊的训练数据：
  - dataroot_GT: /media/disk1/fordata/web_server/huangxiaozheng/blur_gt
  - dataroot_LQ: /media/disk1/fordata/web_server/huangxiaozheng/blur_data
  - dataroot_GT: /media/disk1/fordata/web_server/huangxiaozheng/blur_gt_test
  - dataroot_LQ: /media/disk1/fordata/web_server/huangxiaozheng/blur_data_test
#### 训练代码采用RealSR作为基本框架：
  - 文件目录在29服务器上，/home/web_server/zhouhuanxiang/disk/Real-SR-master；
  - 采用RealSR作为基本框架，具体可参考 https://github.com/Tencent/Real-SR；
  - 在实现上，没有对框架的主要流程进行修改，可直接参考原RealSR的训练、测试方式进行调用；
#### 训练结果：
- 模型路径为，/home/web_server/zhouhuanxiang/disk/Real-SR-master/experiments
- 测试结果路径为，/home/web_server/zhouhuanxiang/disk/Real-SR-master/result
#### 一些尝试：
- RealSR	
  - ./models/modules/RRDBNet_arch.py	
  - ./realsr_kwai
- U^2-Unet	
  - ./models/modules/u2net.py	
  - ./train_u2net_x1_kwai
- UNet + attention	
  - ./models/modules/unet_attn.py	
  - ./realsr_kwai_unet_attn


