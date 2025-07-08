# [yolov11改进系列]使用轻量级反向残差块网络EMO替换backbone的python源码+训练源码

> FL1623863129 于 2025-05-28 07:53:56 发布 阅读量872 收藏 26 点赞数 23 公开
> 文章链接：https://blog.csdn.net/FL1623863129/article/details/148268939

[EMO网络介绍]

### 1. EMO简介

[反向残差块](https://zhida.zhihu.com/search?content_id=251469358&content_type=Article&match_order=1&q=%E5%8F%8D%E5%90%91%E6%AE%8B%E5%B7%AE%E5%9D%97&zhida_source=entity) （Inverted Rsidual Block，IRB）是 [轻量级CNN](https://zhida.zhihu.com/search?content_id=251469358&content_type=Article&match_order=1&q=%E8%BD%BB%E9%87%8F%E7%BA%A7CNN&zhida_source=entity) s的基础架构，但在基于注意力的研究中还没有相应的对应部分。这项工作从统一的视角重新思考高效IRB和 [Transformer](https://zhida.zhihu.com/search?content_id=251469358&content_type=Article&match_order=1&q=Transformer&zhida_source=entity) 的有效组件，将基于CNN的IRB扩展到基于注意力的模型，并抽象出一个用于轻量级模型设计的单残差 [元移动块](https://zhida.zhihu.com/search?content_id=251469358&content_type=Article&match_order=1&q=%E5%85%83%E7%A7%BB%E5%8A%A8%E5%9D%97&zhida_source=entity) （Meta Mobile Block，MMB）。本文推导出了一个现代化的方向残差移动块（Inverted Residual Mobile Block， iRMB），仅使用iRMB构建一个类似 [ResNet](https://zhida.zhihu.com/search?content_id=251469358&content_type=Article&match_order=1&q=ResNet&zhida_source=entity) 的高效模型（EfficientModel， EMO），用于下游任务。

### 2. EMO 创新点

EMO模型基于反向残差块，是一种轻量级CNN的基础架构，同时融合了Transformer的有效组件。通过这种结合，EMO实现了一个统一的视角来处理轻量级模型的设计，创新的将CNN和注意力机制相结合。

EMO的基本原理可以分成以下几个要点：

1. 反向残差块（IRB）的应用：IRB作为轻量级CNN的基础架构，EMO将其扩展到基于注意力的模型

2. 元移动块（MMB）的抽象化：EMO提出了一种新的轻量级设计方法。即单残差的元移动块（MMB），这是从IRB和Transformer的有效组件中抽象出来的

3. 现代反向残差移动块（iRMB）的构建：基于简单但有效的设计标准，EMO推导了iRMB，并以此构建了类似于Reset的高效模型(EMO)

### 3. EMO模型架构

 <img src="./assets/22_1.png" alt="" style="max-height:388px; box-sizing:content-box;" />

左侧：一个抽象统一的元移动块（Meta-Mobile Block），融合了 [多头自注意机制](https://zhida.zhihu.com/search?content_id=251469358&content_type=Article&match_order=1&q=%E5%A4%9A%E5%A4%B4%E8%87%AA%E6%B3%A8%E6%84%8F%E6%9C%BA%E5%88%B6&zhida_source=entity) （Multi-Head Self-Attention）， [前馈网络](https://zhida.zhihu.com/search?content_id=251469358&content_type=Article&match_order=1&q=%E5%89%8D%E9%A6%88%E7%BD%91%E7%BB%9C&zhida_source=entity) （Feed-Forward Network）和反向残差块（Inverted Residual Block）。这个符合模块通过不同的扩展比率和高效的操作符进行具体化

右侧：一个类似于ResNet的EMO模型架构，完全由推导出的iRMB组成。图中突出了EMO模型中微操作组合（如深度可分离卷积，窗口Transformer等）和不同尺度的网络层次，这些都是用于分类，检测和分割任务的。

【yolov11框架介绍】

2024 年 9 月 30 日，Ultralytics 在其活动 YOLOVision 中正式发布了 YOLOv11。YOLOv11 是 YOLO 的最新版本，由美国和西班牙的 Ultralytics 团队开发。YOLO 是一种用于基于图像的人工智能的计算机模

#### Ultralytics YOLO11 概述

YOLO11 是Ultralytics YOLO 系列实时物体检测器的最新版本，以尖端的精度、速度和效率重新定义了可能性。基于先前 YOLO 版本的令人印象深刻的进步，YOLO11 在架构和训练方法方面引入了重大改进，使其成为各种计算机视觉任务的多功能选择。

<img src="./assets/22_2.png" alt="" style="max-height:306px; box-sizing:content-box;" />

#### Key Features 主要特点

- 增强的特征提取：YOLO11采用改进的主干和颈部架构，增强了特征提取能力，以实现更精确的目标检测和复杂任务性能。

- 针对效率和速度进行优化：YOLO11 引入了精致的架构设计和优化的训练管道，提供更快的处理速度并保持准确性和性能之间的最佳平衡。

- 使用更少的参数获得更高的精度：随着模型设计的进步，YOLO11m 在 COCO 数据集上实现了更高的平均精度(mAP)，同时使用的参数比 YOLOv8m 少 22%，从而在不影响精度的情况下提高计算效率。

- 跨环境适应性：YOLO11可以无缝部署在各种环境中，包括边缘设备、云平台以及支持NVIDIA [GPU](https://cloud.tencent.com/product/gpu?from_column=20065&from=20065) 的系统，确保最大的灵活性。

- 支持的任务范围广泛：无论是对象检测、实例分割、图像分类、姿态估计还是定向对象检测 (OBB)，YOLO11 旨在应对各种计算机视觉挑战。

 <img src="./assets/22_3.png" alt="" style="max-height:270px; box-sizing:content-box;" />

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

##### 1. 新增EMO.py实现模块（代码太多，核心模块源码请参考改进步骤.docx）然后在同级目录下面创建一个__init___.py文件写代码

from .EMO import *

##### 2. 文件修改步骤

**修改tasks.py文件** 

**创建模型配置文件** 

yolo11-EMO.yaml内容如下：

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
 
# 我提供了版本分别是对应是 ['EMO_1M', 'EMO_2M', 'EMO_5M', 'EMO_6M']
# 其中n是对应yolo的版本通道放缩 large 和 small 是模型官方本身自带的版本
# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, EMO_1M, [0.25]] # 0-4 P1/2 这里是四层
  # 注意args位置的参数对应模型的通道放缩系数width在上面scales位置, 假设你用yolov11n那么可以设置0.25 如果你用yolov11s可以设置0.5
  - [-1, 1, SPPF, [1024, 5]] # 5
  - [-1, 2, C2PSA, [1024]] # 6
 
# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 3], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 9
 
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 12 (P3/8-small)
 
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 15 (P4/16-medium)
 
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 6], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 18 (P5/32-large)
 
  - [[12, 15, 18], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

##### 3. 验证集成

使用新建的yaml配置文件启动训练任务：

```cobol
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('yolo11-EMO.yaml')  # build from YAML and transfer weights
        # Train the model
    results = model.train(data='coco128.yaml',epochs=100, imgsz=640, batch=8, device=0, workers=1, save=True,resume=False)
```

成功集成后，训练日志中将显示EMO模块的初始化信息，表明已正确加载到模型中。

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/fcf82f9dda6d46c8aac14372d3a3f12b.jpeg">​</div>

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
│   ├── yolo11-EMO.yaml
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