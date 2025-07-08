# [yolov11改进系列]基于yolov11引入自集成注意力机制SEAM解决遮挡问题的python源码+训练源码

> FL1623863129 于 2025-06-03 14:46:11 发布 阅读量757 收藏 16 点赞数 26 公开
> 文章链接：https://blog.csdn.net/FL1623863129/article/details/148401428

【SEAM注意力机制介绍】

本文给大家带来的改进机制是由YOLO-Face提出能够改善物体遮挡检测的注意力机制SEAM， **SEAM（Spatially Enhanced Attention Module）** 注意力网络模块旨在补偿被遮挡面部的响应损失，通过增强未遮挡面部的响应来实现这一目标，其希望通过学习遮挡面和未遮挡面之间的关系来改善遮挡情况下的损失从而达到改善物体遮挡检测的效果。

 <img src="./assets/47_1.png" alt="" style="max-height:684px; box-sizing:content-box;" />

### 1.1 遮挡改进

本文重点介绍 **遮挡改进** ，其主要体现在两个方面： **注意力网络模块（SEAM）** 和 **排斥损失（ [Repulsion Loss](https://zhida.zhihu.com/search?content_id=244606480&content_type=Article&match_order=1&q=Repulsion+Loss&zhida_source=entity) ）** 。

**1. SEAM模块：SEAM（Spatially Enhanced Attention Module）** 注意力网络模块旨在补偿被遮挡面部的响应损失，通过增强未遮挡面部的响应来实现这一目标。SEAM模块通过深度可分离卷积和残差连接的组合来实现，其中深度可分离卷积按通道进行操作，虽然可以学习不同通道的重要性并减少参数量，但忽略了通道间的信息关系。为了弥补这一损失，不同深度卷积的输出通过点对点（1x1）卷积组合。然后使用两层全连接网络融合每个通道的信息，以增强所有通道之间的联系。这种模型希望通过学习遮挡面和未遮挡面之间的关系，来弥补遮挡情况下的损失。

**2. 排斥损失（Repulsion Loss）** ：一种设计来处理面部遮挡问题的损失函数。具体来说，排斥损失被分为两部分：RepGT和RepBox。RepGT的功能是使当前的边界框尽可能远离周围的真实边界框，而RepBox的目的是使预测框尽可能远离周围的预测框，从而减少它们之间的IOU，以避免某个预测框被NMS抑制，从而属于两个面部。

---

### 1.2 **SEAM模块** 

下图展示了 **SEAM（Separated and Enhancement Attention Module）的架构** 以及 **[CSMM](https://zhida.zhihu.com/search?content_id=244606480&content_type=Article&match_order=1&q=CSMM&zhida_source=entity) （Channel and Spatial Mixing Module）的结构** 。



 <img src="./assets/47_2.png" alt="" style="max-height:654px; box-sizing:content-box;" />

**左侧** 是SEAM的整体架构，包括三个不同尺寸（patch-6、patch-7、patch-8）的CSMM模块。这些模块的输出进行平均池化，然后通过通道扩展（Channel exp）操作，最后相乘以提供增强的特征表示。 **右侧** 是CSMM模块的详细结构，它通过不同尺寸的patch来利用多尺度特征，并使用深度可分离卷积来学习空间维度和通道之间的相关性。模块包括了以下元素：

（a）Patch Embedding：对输入的patch进行嵌入。
（b） [GELU](https://zhida.zhihu.com/search?content_id=244606480&content_type=Article&match_order=1&q=GELU&zhida_source=entity) ：Gaussian Error Linear Unit，一种激活函数。
（c） [BatchNorm](https://zhida.zhihu.com/search?content_id=244606480&content_type=Article&match_order=1&q=BatchNorm&zhida_source=entity) ：批量归一化，用于加速训练过程并提高性能。
（d） [Depthwise Convolution](https://zhida.zhihu.com/search?content_id=244606480&content_type=Article&match_order=1&q=Depthwise+Convolution&zhida_source=entity) ：深度可分离卷积，对每个输入通道分别进行卷积操作。
（f） [Pointwise Convolution](https://zhida.zhihu.com/search?content_id=244606480&content_type=Article&match_order=1&q=Pointwise+Convolution&zhida_source=entity) ：逐点卷积，其使用1x1的卷积核来融合深度可分离卷积的特征。

这种模块设计旨在 **通过对空间维度和通道的细致处理** ，从而增强网络对遮挡面部特征的注意力和捕捉能力。通过综合利用多尺度特征和深度可分离卷积，CSMM在保持计算效率的同时，提高了特征提取的精确度。这对于面部检测尤其重要，因为面部特征的大小、形状和遮挡程度可以在不同情况下大相径庭。通过SEAM和CSMM，YOLO-FaceV2提高了模型对复杂场景中各种面部特征的识别能力。

【yolov11框架介绍】

2024 年 9 月 30 日，Ultralytics在其活动 YOLOVision 中正式发布了 YOLOv11。YOLOv11 是 YOLO 的最新版本，由美国和西班牙的 Ultralytics 团队开发。YOLO 是一种用于基于图像的人工智能的计算机模

#### Ultralytics YOLO11 概述

YOLO11 是Ultralytics YOLO 系列实时物体检测器的最新版本，以尖端的精度、速度和效率重新定义了可能性。基于先前 YOLO 版本的令人印象深刻的进步，YOLO11 在架构和训练方法方面引入了重大改进，使其成为各种计算机视觉任务的多功能选择。

<img src="./assets/47_3.png" alt="" style="max-height:306px; box-sizing:content-box;" />

#### Key Features 主要特点

- 增强的特征提取：YOLO11采用改进的主干和颈部架构，增强了特征提取能力，以实现更精确的目标检测和复杂任务性能。

- 针对效率和速度进行优化：YOLO11 引入了精致的架构设计和优化的训练管道，提供更快的处理速度并保持准确性和性能之间的最佳平衡。

- 使用更少的参数获得更高的精度：随着模型设计的进步，YOLO11m 在 COCO 数据集上实现了更高的平均精度(mAP)，同时使用的参数比 YOLOv8m 少 22%，从而在不影响精度的情况下提高计算效率。

- 跨环境适应性：YOLO11可以无缝部署在各种环境中，包括边缘设备、云平台以及支持NVIDIA [GPU](https://cloud.tencent.com/product/gpu?from_column=20065&from=20065) 的系统，确保最大的灵活性。

- 支持的任务范围广泛：无论是对象检测、实例分割、图像分类、姿态估计还是定向对象检测 (OBB)，YOLO11 旨在应对各种计算机视觉挑战。

 <img src="./assets/47_4.png" alt="" style="max-height:270px; box-sizing:content-box;" />

​​​

##### 与之前的版本相比，Ultralytics YOLO11 有哪些关键改进？

Ultralytics YOLO11 与其前身相比引入了多项重大进步。主要改进包括：

- 增强的特征提取：YOLO11采用改进的主干和颈部架构，增强了特征提取能力，以实现更精确的目标检测。

- 优化的效率和速度：精细的架构设计和优化的训练管道可提供更快的处理速度，同时保持准确性和性能之间的平衡。

- 使用更少的参数获得更高的精度：YOLO11m 在 COCO 数据集上实现了更高的平均精度(mAP)，参数比 YOLOv8m 少 22%，从而在不影响精度的情况下提高计算效率。

- 跨环境适应性：YOLO11可以跨各种环境部署，包括边缘设备、云平台和支持NVIDIA GPU的系统。

- 支持的任务范围广泛：YOLO11 支持多种计算机视觉任务，例如对象检测、实例分割、图像分类、姿态估计和定向对象检测 (OBB)

【测试环境】

windows10 x64

ultralytics==8.3.0

torch==2.3.1

【改进流程】

##### 1. 新增SEAM.py实现模块（代码太多，核心模块源码请参考改进步骤.docx）然后在同级目录下面创建一个__init___.py文件写代码

from .SEAM import *

##### 2. 文件修改步骤

**修改tasks.py文件** 

**创建模型配置文件** 

yolo11-SEAM.yaml内容如下：

```cobol
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLO11 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
 
# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs
 
# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10
 
# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13
 
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)
  - [-1, 1, SEAM, []] # 17 (P3/8-small)  小目标检测层输出位置增加注意力机制
 
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 20 (P4/16-medium)
  - [-1, 1, SEAM, []] # 21 (P4/16-medium) 中目标检测层输出位置增加注意力机制
 
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 24 (P5/32-large)
  - [-1, 1, SEAM, []] # 25 (P5/32-large) 大目标检测层输出位置增加注意力机制
 
  # 具体在那一层用注意力机制可以根据自己的数据集场景进行选择。
  # 如果你自己配置注意力位置注意from[17, 21, 25]位置要对应上对应的检测层！
  - [[17, 21, 25], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

##### 3. 验证集成

git搜futureflsl/yolo-improve获取源码，然后使用新建的yaml配置文件启动训练任务：

```cobol
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('yolo11-SEAM.yaml')  # build from YAML and transfer weights
        # Train the model
    results = model.train(data='coco128.yaml',epochs=100, imgsz=640, batch=8, device=0, workers=1, save=True,resume=False)
```

成功集成后，训练日志中将显示SEAM模块的初始化信息，表明已正确加载到模型中。

<div style="text-align:center;">​<img alt="" src="https://i-blog.csdnimg.cn/direct/a4adbfe029db438c81d20f8dfeb71093.jpeg">​</div>

【训练说明】

第一步：首先安装好yolov11必要模块，可以参考yolov11框架安装流程，然后卸载官方版本pip uninstall ultralytics，最后安装改进的源码pip install .
第二步：将自己数据集按照dataset文件夹摆放，要求文件夹名字都不要改变
第三步：分别打开train.py,coco128.yaml和模型参数yaml文件修改必要的参数，最后执行python train.py即可训练

【提供文件】

```cobol
├── [官方源码]ultralytics-8.3.0.zip
├── train/
│   ├── coco128.yaml
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   │   ├── firc_pic_1.jpg
│   │   │   │   ├── firc_pic_10.jpg
│   │   │   │   ├── firc_pic_11.jpg
│   │   │   │   ├── firc_pic_12.jpg
│   │   │   │   ├── firc_pic_13.jpg
│   │   │   ├── labels/
│   │   │   │   ├── classes.txt
│   │   │   │   ├── firc_pic_1.txt
│   │   │   │   ├── firc_pic_10.txt
│   │   │   │   ├── firc_pic_11.txt
│   │   │   │   ├── firc_pic_12.txt
│   │   │   │   ├── firc_pic_13.txt
│   │   └── val/
│   │       ├── images/
│   │       │   ├── firc_pic_100.jpg
│   │       │   ├── firc_pic_81.jpg
│   │       │   ├── firc_pic_82.jpg
│   │       │   ├── firc_pic_83.jpg
│   │       │   ├── firc_pic_84.jpg
│   │       ├── labels/
│   │       │   ├── firc_pic_100.txt
│   │       │   ├── firc_pic_81.txt
│   │       │   ├── firc_pic_82.txt
│   │       │   ├── firc_pic_83.txt
│   │       │   ├── firc_pic_84.txt
│   ├── train.py
│   ├── yolo11-SEAM.yaml
│   └── 训练说明.txt
├── [改进源码]ultralytics-8.3.0.zip
├── 改进原理.docx
└── 改进流程.docx
```

【常见问题汇总】
问：为什么我训练的模型epoch显示的map都是0或者map精度很低?
回答：由于源码改进过，因此不能直接从官方模型微调，而是从头训练，这样学习特征能力会很弱，需要训练很多epoch才能出现效果。此外由于改进的源码框架并不一定能够保证会超过官方精度，而且也有可能会存在远远不如官方效果，甚至精度会很低。这说明改进的框架并不能取得很好效果。所以说对于框架改进只是提供一种可行方案，至于改进后能不能取得很好map还需要结合实际训练情况确认，当然也不排除数据集存在问题，比如数据集比较单一，样本分布不均衡，泛化场景少，标注框不太贴合标注质量差，检测目标很小等等原因
【重要说明】
我们只提供改进框架一种方案，并不保证能够取得很好训练精度，甚至超过官方模型精度。因为改进框架，实际是一种比较复杂流程，包括框架原理可行性，训练数据集是否合适，训练需要反正验证以及同类框架训练结果参数比较，这个是十分复杂且漫长的过程。