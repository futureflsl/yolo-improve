# [yolov11改进系列]基于yolov11使用极简主义网络VanillaNet替换backbone的python源码+训练源码

> FL1623863129 于 2025-06-02 21:04:56 发布 阅读量1.1k 收藏 10 点赞数 35 公开
> 文章链接：https://blog.csdn.net/FL1623863129/article/details/148386687

【VanillaNet介绍】

一、VanillaNet概述
1.1 摘要
基础模型的核心是“更多不同”的理念，计算机视觉和自然语言处理方面的出色表现就是例证。然而，Transformer模型的优化和固有复杂性的挑战要求范式向简单性转变。在本文中，我们介绍了VanillaNET，这是一种设计优雅的神经网络架构。通过避免高深度、shortcuts和自注意力等复杂操作，VanillaNet简洁明了但功能强大。每一层都经过精心设计，非线性激活函数在训练后被修剪以恢复原始架构。VanillaNet克服了固有复杂性的挑战，使其成为资源受限环境的理想选择。其易于理解和高度简化的架构为高效部署开辟了新的可能性。广泛的实验表明，VanillaNet提供的性能与著名的深度神经网络和vision transformers相当，展示了深度学习中极简主义的力量。VanillaNet的这一富有远见的旅程具有重新定义景观和挑战基础模型现状的巨大潜力，为优雅有效的模型设计开辟了一条新道路

1.2 VanillaNet结构
在过去的几十年里，研究人员在神经网络的基本设计上达成了一些共识。大多数最先进的图像分类网络架构应该由三部分组成：一个主干块，用于将输入图像从3个通道转换为多个通道，并进行下采样，一个学习有用的信息主题，一个全连接层分类输出。主体通常有四个阶段，每个阶段都是通过堆叠相同的块来派生的。在每个阶段之后，特征的通道将扩展，而高度和宽度将减小。不同的网络利用和堆叠不同种类的块来构建深度模型。

尽管现有的深度网络取得了成功，但它们利用大量复杂层来为以下任务提取高级特征。例如，著名的ResNet需要34或50个带shortcat的层才能在ImageNet上实现超过70%的top-1精度。Vit的基础版本由62层组成，因为自注意力中的K、Q、V需要多层来计算。

随着AI芯片雨来越大，神经网络推理速度的瓶颈不再是FLOPs或参数，因为现代GPU可以很容易地进行并行计算。相比之下，它们复杂的设计和较大的深度阻碍了它们的速度。为此我们提出了Vanilla网络，即VanillaNet，其框架图如图一所示。我们遵循流行的神经网络设计，包括主干、主体和全连接层。与现有的深度网络不同，我们在每个阶段只使用一层，以建立一个尽可能少的层的极其简单的网络。

 <img src="./assets/44_1.png" alt="" style="max-height:788px; box-sizing:content-box;" />

这里我们详细展示了VanillaNet的架构，以6层为例。对于主干，我们使用步长为4的4 × 4 × 3 × C 4 \times 4 \times 3 \times C4×4×3×C卷积层，遵循流行设置，将具有3个通道的图像映射到具有C个通道的特征。在1、2和3阶段，使用步幅为2的最大池化层来减小尺寸和特征图，并将通道数增加2。在第4阶段，我们不增加通道数，因为它使用平均池化层。最后一层是全连接层，输出分类结果。

每个卷积核的内核大小为1 × 1 1 \times 11×1，因为我们的目标是在保留特征图信息的同时对每一层使用最小的计算成本。在每个1 × 1 1 \times 11×1卷积层之后应用激活函数。为了简化网络的训练过程，还在每一层之后添加了批量归一化。VanillaNet没有shortcut，因为我们凭经验发现添加shortcut几乎没有提高性能。

这也带来的另一个好处，即所提出的架构非常容易实现，因为没有分支和额外的块，例如squeeze和excitation block。虽然VanillaNet的体系结构简单且相对较浅，但其弱非线性导致性能受到限制，因此，我们提出了一系列技术来解决该问题。

1.3 结论
本文充分研究了建立高性能神经网络的可行性，但没有复杂的架构，如快捷方式、高深度和注意层，这体现了设计向简单和优雅的范式转变。我们为VanillaNets提出了一种深度训练策略和系列激活函数，以增强其在训练和测试过程中的非线性并提高其性能。大规模图像分类数据集的实验结果表明，VanillaNet的性能与著名的深度神经网络和视觉转换器相当，从而突出了极简主义在深度学习中的潜力。我们将进一步探索更好的参数分配，以获得高性能的高效VanillaNet架构。总之，我们证明可以使用非常简洁的架构与最先进的深度网络和视觉转换器实现可比的性能，这将在未来释放普通卷积网络的潜力。

【yolov11框架介绍】

2024 年 9 月 30 日，Ultralytics 在其活动 YOLOVision 中正式发布了 YOLOv11。YOLOv11 是 YOLO 的最新版本，由美国和西班牙的 Ultralytics 团队开发。YOLO 是一种用于基于图像的人工智能的计算机模

#### Ultralytics YOLO11 概述

YOLO11 是Ultralytics YOLO 系列实时物体检测器的最新版本，以尖端的精度、速度和效率重新定义了可能性。基于先前 YOLO 版本的令人印象深刻的进步，YOLO11 在架构和训练方法方面引入了重大改进，使其成为各种计算机视觉任务的多功能选择。

<img src="./assets/44_2.png" alt="" style="max-height:306px; box-sizing:content-box;" />

#### Key Features 主要特点

- 增强的特征提取：YOLO11采用改进的主干和颈部架构，增强了特征提取能力，以实现更精确的目标检测和复杂任务性能。

- 针对效率和速度进行优化：YOLO11 引入了精致的架构设计和优化的训练管道，提供更快的处理速度并保持准确性和性能之间的最佳平衡。

- 使用更少的参数获得更高的精度：随着模型设计的进步，YOLO11m 在 COCO 数据集上实现了更高的平均精度(mAP)，同时使用的参数比 YOLOv8m 少 22%，从而在不影响精度的情况下提高计算效率。

- 跨环境适应性：YOLO11可以无缝部署在各种环境中，包括边缘设备、云平台以及支持NVIDIA [GPU](https://cloud.tencent.com/product/gpu?from_column=20065&from=20065) 的系统，确保最大的灵活性。

- 支持的任务范围广泛：无论是对象检测、实例分割、图像分类、姿态估计还是定向对象检测 (OBB)，YOLO11 旨在应对各种计算机视觉挑战。

 <img src="./assets/44_3.png" alt="" style="max-height:270px; box-sizing:content-box;" />

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

RTX2070显卡8GB，使用改进源码推荐显存>=6GB，显卡型号不限

【改进流程】

##### 1. 新增vanillanet.py实现模块（代码太多，核心模块源码请参考改进步骤.docx）然后在同级目录下面创建一个__init___.py文件写代码

from .vanillanet import *

##### 2. 文件修改步骤

**修改tasks.py文件** 

**创建模型配置文件** 

yolo11-vanillanet.yaml内容如下：

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
 
# 下面 [-1, 1, vanillanet_5, [0.25]] 参数位置的0.25是通道放缩的系数, YOLOv11N是0.25 YOLOv11S是0.5 YOLOv11M是1. YOLOv11l是1 YOLOv11是1.5大家根据自己训练的YOLO版本设定即可.
# 本文支持版本有 vanillanet_5, vanillanet_6, vanillanet_7, vanillanet_8, vanillanet_9, vanillanet_10, vanillanet_11, vanillanet_12, vanillanet_13, vanillanet_13_x1_5, vanillanet_13_x1_5_ada_pool
# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, vanillanet_5, [0.25]] # 0-4 P1/2 这里是四层
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

git搜futureflsl/yolo-improve获取源码，然后使用新建的yaml配置文件启动训练任务：

```cobol
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('yolo11-vanillanet.yaml')  # build from YAML and transfer weights
        # Train the model
    results = model.train(data='coco128.yaml',epochs=100, imgsz=640, batch=8, device=0, workers=1, save=True,resume=False)
```

成功集成后，训练日志中将显示vanillanet模块的初始化信息，表明已正确加载到模型中。

<div style="text-align:center;">​<img alt="" src="https://i-blog.csdnimg.cn/direct/053a31614d72458ba2454646e0bb61c1.jpeg">​</div>

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
│   ├── yolo11-vanillanet.yaml
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