# [yolov11改进系列]基于yolov11引入自注意力与卷积混合模块ACmix提高FPS+检测效率python源码+训练源码

> FL1623863129 已于 2025-05-26 07:36:09 修改 阅读量1k 收藏 27 点赞数 14 公开
> 文章链接：https://blog.csdn.net/FL1623863129/article/details/148217649

[ACmix的框架原理]

 <img src="./assets/11_1.png" alt="" style="max-height:243px; box-sizing:content-box;" />

 <img src="./assets/11_2.png" alt="" style="max-height:309px; box-sizing:content-box;" />

1.1 ACMix的基本原理

ACmix是一种混合模型，结合了自注意力机制和卷积运算的优势。它的核心思想是，传统卷积操作和自注意力模块的大部分计算都可以通过1x1的卷积来实现。ACmix首先使用1x1卷积对输入特征图进行投影，生成一组中间特征，然后根据不同的范式，即自注意力和卷积方式，分别重用和聚合这些中间特征。这样，ACmix既能利用自注意力的全局感知能力，又能通过卷积捕获局部特征，从而在保持较低计算成本的同时，提高模型的性能。

ACmix模型的主要改进机制可以分为以下两点：

1. 自注意力和卷积的整合：将自注意力和卷积技术融合，实现两者优势的结合。 2. 运算分解与重构：通过分解自注意力和卷积中的运算，重构为1×1卷积形式，提高了运算效率。

1.1.1 自注意力和卷积的整合

文章中指出，自注意力和卷积的整合通过以下方式实现：

特征分解：自注意力机制的查询（query）、键（key）、值（value）与卷积操作通过1x1卷积进行特征分解。 运算共享：卷积和自注意力共享相同的1x1卷积运算，减少了重复的计算量。 特征融合：在ACmix模型中，卷积和自注意力生成的特征通过求和操作进行融合，加强了模型的特征提取能力。 模块化设计：通过模块化设计，ACmix可以灵活地嵌入到不同的网络结构中，增强网络的表征能力。





 <img src="./assets/11_3.png" alt="" style="max-height:433px; box-sizing:content-box;" />

这张图片展示了ACmix中的主要概念，它比较了卷积、自注意力和ACmix各自的结构和计算复杂度。图中：

(a) 卷积：展示了标准卷积操作，包含一个

<img src="./assets/11_4.png" alt="" style="max-height:16px; box-sizing:content-box;" />

的1x1卷积，表示卷积核大小和卷积操作的聚合。

(b) 自注意力：展示了自注意力机制，它包含三个头部的1x1卷积，代表多头注意力机制中每个头部的线性变换，以及自注意力聚合。

(c) ACmix（我们的方法）：结合了卷积和自注意力聚合，其中1x1卷积在两者之间共享，旨在减少计算开销并整合轻量级的聚合操作。

整体上，ACmix旨在通过共享计算资源（1x1卷积）并结合两种不同的聚合操作，以优化特征通道上的计算复杂度。



1.1.2 运算分解与重构

在ACmix中，运算分解与重构的概念是指将传统的卷积运算和自注意力运算拆分，并重新构建为更高效的形式。这主要通过以下步骤实现：

分解卷积和自注意力：将标准的卷积核分解成多个1×1卷积核，每个核处理不同的特征子集，同时将自注意力机制中的查询（query）、键（key）和值（value）的生成也转换为1×1卷积操作。 重构为混合模块：将分解后的卷积和自注意力运算重构成一个统一的混合模块，既包含了卷积的空间特征提取能力，也融入了自注意力的全局信息聚合功能。 提高运算效率：这种分解与重构的方法减少了冗余计算，提高了运算效率，同时降低了模型的复杂度。



 <img src="./assets/11_5.png" alt="" style="max-height:560px; box-sizing:content-box;" />

这张图片展示了ACmix提出的混合模块的结构。图示包含了：

(a) 卷积：3x3卷积通过1x1卷积的方式被分解，展示了特征图的转换过程。

(b)自注意力：输入特征先转换成查询（query）、键（key）和值（value），使用1x1卷积实现，并通过相似度匹配计算注意力权重。

(c) ACmix：结合了(a)和(b)的特点，在第一阶段使用三个1x1卷积对输入特征图进行投影，在第二阶段将两种路径得到的特征相加，作为最终输出。

右图显示了ACmix模块的流程，强调了两种机制的融合并提供了每个操作块的计算复杂度。

【yolov11框架介绍】

2024 年 9 月 30 日，Ultralytics在其活动 YOLOVision 中正式发布了 YOLOv11。YOLOv11 是 YOLO 的最新版本，由美国和西班牙的 Ultralytics 团队开发。YOLO 是一种用于基于图像的人工智能的计算机模

#### Ultralytics YOLO11 概述

YOLO11 是Ultralytics YOLO 系列实时物体检测器的最新版本，以尖端的精度、速度和效率重新定义了可能性。基于先前 YOLO 版本的令人印象深刻的进步，YOLO11 在架构和训练方法方面引入了重大改进，使其成为各种计算机视觉任务的多功能选择。

<img src="./assets/11_6.png" alt="" style="max-height:306px; box-sizing:content-box;" />

#### Key Features 主要特点

- 增强的特征提取：YOLO11采用改进的主干和颈部架构，增强了特征提取能力，以实现更精确的目标检测和复杂任务性能。

- 针对效率和速度进行优化：YOLO11 引入了精致的架构设计和优化的训练管道，提供更快的处理速度并保持准确性和性能之间的最佳平衡。

- 使用更少的参数获得更高的精度：随着模型设计的进步，YOLO11m 在 COCO 数据集上实现了更高的平均精度(mAP)，同时使用的参数比 YOLOv8m 少 22%，从而在不影响精度的情况下提高计算效率。

- 跨环境适应性：YOLO11可以无缝部署在各种环境中，包括边缘设备、云平台以及支持NVIDIA [GPU](https://cloud.tencent.com/product/gpu?from_column=20065&from=20065) 的系统，确保最大的灵活性。

- 支持的任务范围广泛：无论是对象检测、实例分割、图像分类、姿态估计还是定向对象检测 (OBB)，YOLO11 旨在应对各种计算机视觉挑战。

 <img src="./assets/11_7.png" alt="" style="max-height:270px; box-sizing:content-box;" />

​

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

##### 1. 新增ACmix.py实现骨干网络（代码太多，核心模块源码请参考改进步骤.docx）

##### 2. 文件修改步骤

**修改tasks.py文件** 

**创建模型配置文件** 

yolo11-ACmix.yaml内容如下：

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
  - [-1, 1, ACmix, []] # 17 (P3/8-small)  小目标检测层输出位置增加注意力机制
 
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 20 (P4/16-medium)
  - [-1, 1, ACmix, []] # 21 (P4/16-medium) 中目标检测层输出位置增加注意力机制
 
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 24 (P5/32-large)
  - [-1, 1, ACmix, []] # 25 (P5/32-large) 大目标检测层输出位置增加注意力机制
 
  # 注意力机制我这里其实是添加了三个但是实际一般生效就只添加一个就可以了，所以大家可以自行注释来尝试, 上面三个仅建议大家保留一个， 但是from位置要对齐.
  # 具体在那一层用注意力机制可以根据自己的数据集场景进行选择。
  # 如果你自己配置注意力位置注意from[17, 21, 25]位置要对应上对应的检测层！
  - [[17, 21, 25], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

##### 3. 验证集成

使用新建的yaml配置文件启动训练任务：

```cobol
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('yolo11-ACmix.yaml')  # build from YAML and transfer weights
        # Train the model
    results = model.train(data='coco128.yaml',epochs=100, imgsz=640, batch=8, device=0, workers=1, save=True,resume=False)
```

成功集成后，训练日志中将显示ACmix模块的初始化信息，表明已正确加载到模型中。

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/2381f3b5816a4edb83b74966be0cfa93.jpeg"></div>

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
│   ├── yolo11-ACmix.yaml
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