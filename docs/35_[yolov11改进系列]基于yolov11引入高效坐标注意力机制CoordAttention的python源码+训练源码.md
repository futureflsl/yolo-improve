# [yolov11改进系列]基于yolov11引入高效坐标注意力机制CoordAttention的python源码+训练源码

> FL1623863129 于 2025-05-31 07:37:42 发布 阅读量745 收藏 21 点赞数 22 公开
> 文章链接：https://blog.csdn.net/FL1623863129/article/details/148348371

【CoordAttention介绍】

在轻量级网络上的研究表明，通道注意力会给模型带来比较显著的性能提升，但是通道注意力通常会忽略对生成空间选择性注意力图非常重要的位置信息。因此，新加坡国立大学的提出了一种为轻量级网络设计的新的注意力机制，该机制将位置信息嵌入到了通道注意力中，称为Coordinate Attention（下文也称CA），该论文收录于CVPR2021。不同于通道注意力将输入通过2D全局池化转化为单个特征向量，CoordAttention将通道注意力分解为两个沿着不同方向聚合特征的1D特征编码过程。这样的好处是可以沿着一个空间方向捕获长程依赖，沿着另一个空间方向保留精确的位置信息。然后，将生成的特征图分别编码，形成一对方向感知和位置敏感的特征图，它们可以互补地应用到输入特征图来增强感兴趣的目标的表示。CA简单灵活且高效，可以插入经典的轻量级网络（如MobileNetV2）在几乎不带来额外计算开销的前提下，提升网络的精度。实验表明，CoordAttention不仅仅对于分类任务有不错的提高，对目标检测、实例分割这类密集预测的任务，效果提升更加明显。

论文地址：http://arxiv.org/abs/2103.02907

论文源码https://github.com/Andrew-Qibin/CoordAttention

模型结构
CA是一种高效的注意力机制，通过将位置信息嵌入到通道注意力中，使得轻量级网络能够在更大的区域上进行注意力，同时避免了产生大量的计算开销。为了缓解2D全局池化造成的位置信息丢失，作者将通道注意力分解为两个并行的1D特征编码过程，有效地将空间坐标信息整合到生成的注意图中。更具体来说，作者利用两个一维全局池化操作分别将垂直和水平方向的输入特征聚合为两个独立的方向感知特征图。然后，这两个嵌入特定方向信息的特征图分别被编码为两个注意力图，每个注意力图都捕获了输入特征图沿着一个空间方向的长程依赖。因此，位置信息就被保存在生成的注意力图里了，两个注意力图接着被乘到输入特征图上来增强特征图的表示能力。SEAttention、CBAM以及CA结构如下所示。

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/d94dba2083ee44cca8da775fbacc73c6.jpeg"></div>


一个coordinate attention模块可以看作一个用来增强特征表示能力的计算单元。它可以将任何中间张量X作为输入并输出一个有着增强的表示能力的同样尺寸的输出Y。CA模块通过精确的位置信息对通道关系和长程依赖进行编码，类似SE模块，也分为两个步骤：坐标信息嵌入（coordinate information embedding）和坐标注意力生成（coordinate attention generation）。首先，我们来看坐标信息嵌入这部分。全局池化常用于通道注意力中来全局编码空间信息为通道描述符，因此难以保存位置信息。为了促进注意力模块能够捕获具有精确位置信息的空间长程依赖，作者将全局池化分解为一对一维特征编码操作。具体而言，对输入X，先使用尺寸( H , 1)和(1, W)的池化核沿着水平坐标方向和竖直坐标方向对每个通道进行编码。接着，为了更好地利用上面coordinate information embedding模块产生的具有全局感受野并拥有精确位置信息的表示，设计了coordinate attention generation操作，它生成注意力图。

【yolov11框架介绍】

2024 年 9 月 30 日，Ultralytics在其活动 YOLOVision 中正式发布了 YOLOv11。YOLOv11 是 YOLO 的最新版本，由美国和西班牙的 Ultralytics 团队开发。YOLO 是一种用于基于图像的人工智能的计算机模

#### Ultralytics YOLO11 概述

YOLO11 是Ultralytics YOLO 系列实时物体检测器的最新版本，以尖端的精度、速度和效率重新定义了可能性。基于先前 YOLO 版本的令人印象深刻的进步，YOLO11 在架构和训练方法方面引入了重大改进，使其成为各种计算机视觉任务的多功能选择。

<img src="./assets/35_2.png" alt="" style="max-height:306px; box-sizing:content-box;" />

#### Key Features 主要特点

- 增强的特征提取：YOLO11采用改进的主干和颈部架构，增强了特征提取能力，以实现更精确的目标检测和复杂任务性能。

- 针对效率和速度进行优化：YOLO11 引入了精致的架构设计和优化的训练管道，提供更快的处理速度并保持准确性和性能之间的最佳平衡。

- 使用更少的参数获得更高的精度：随着模型设计的进步，YOLO11m 在 COCO 数据集上实现了更高的平均精度(mAP)，同时使用的参数比 YOLOv8m 少 22%，从而在不影响精度的情况下提高计算效率。

- 跨环境适应性：YOLO11可以无缝部署在各种环境中，包括边缘设备、云平台以及支持NVIDIA [GPU](https://cloud.tencent.com/product/gpu?from_column=20065&from=20065) 的系统，确保最大的灵活性。

- 支持的任务范围广泛：无论是对象检测、实例分割、图像分类、姿态估计还是定向对象检测 (OBB)，YOLO11 旨在应对各种计算机视觉挑战。

 <img src="./assets/35_3.png" alt="" style="max-height:270px; box-sizing:content-box;" />

​​

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

##### 1. 新增CoordAtt.py实现模块（代码太多，核心模块源码请参考改进步骤.docx）然后在同级目录下面创建一个__init___.py文件写代码

from .CoordAtt import *

##### 2. 文件修改步骤

**修改tasks.py文件** 

**创建模型配置文件** 

yolo11-CoordAtt.yaml内容如下：

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
 
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)
 
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)
  - [-1, 1, CoordAtt, []]  # 23 
  - [[16, 19, 23], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

##### 3. 验证集成

使用新建的yaml配置文件启动训练任务：

```cobol
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('yolo11-CoordAtt.yaml')  # build from YAML and transfer weights
        # Train the model
    results = model.train(data='coco128.yaml',epochs=100, imgsz=640, batch=8, device=0, workers=1, save=True,resume=False)
```

成功集成后，训练日志中将显示CoordAtt模块的初始化信息，表明已正确加载到模型中。

<div style="text-align:center;">​<img alt="" src="https://i-blog.csdnimg.cn/direct/1eb6e86d14d54f61b822d3a4e64efbcb.jpeg"></div>

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
│   ├── yolo11-CoordAtt.yaml
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