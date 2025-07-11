# 基于YOLO算改进法系列汇总

> 置顶 FL1623863129 已于 2025-07-08 10:48:34 修改 阅读量549 收藏 21 点赞数 21 公开
> 文章链接：https://blog.csdn.net/FL1623863129/article/details/148627153

csdn博文YOLO改进系列：https://blog.csdn.net/fl1623863129/category_12975070.html?spm=1001.2014.3001.5482

| 项目名称 | 详情介绍 |
|:---:|:---:|
| [yolov11改进系列]基于yolov11引入单头自注意力机制SHSA的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148509469)  |
| [yolov11改进系列]基于yolov11引入频率感知特征融合模块FreqFusion的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148500432)  |
| [yolov11改进系列]基于yolov11改进检测头引入DynamicHead的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148499550)  |
| [yolov11改进系列]基于yolov11改进检测头引入分布移位卷积DSConvHead的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148497827)  |
| [yolov11改进系列]基于yolov11使用CPA-Enhancer自适应增强器替换backbone提高低照度目标的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148495078)  |
| [yolov11改进系列]基于yolov11融合改进检测头特征融合模块AFPN的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148494399)  |
| [yolov11改进系列]基于yolov11融合改进检测头AFPN4的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148494019)  |
| [yolov11改进系列]基于yolov11引入Haar小波下采样Down_wt卷积减少信息丢失的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148492053)  |
| [yolov11改进系列]基于yolov11使用图像去雾网络AOD-PONO-Net替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148491408)  |
| [yolov11改进系列]基于yolov11使用可逆列网络RevColV1替换backbone用于提高小目标检测能力的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148486612)  |
| [yolov11改进系列]基于yolov11引入轻量级下采样ContextGuided的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148486217)  |
| [yolov11改进系列]基于yolov11引入多层次特征融合模块SDI用于提升小目标识别能力的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148463167)  |
| [yolov11改进系列]基于yolov11使用轻量级新主干RepViT替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148413928)  |
| [yolov11改进系列]基于yolov11引入卷积KANConv含九种不同激活函数的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148462824)  |
| [yolov11改进系列]基于yolov11引入在线重参数化卷积OREPA用于推理加速的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148459419)  |
| [yolov11改进系列]基于yolov11使用SwinTransformer替换backbone用于提高多尺度特征提取能力的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148447275)  |
| [yolov11改进系列]基于yolov11引入注意力机制SENetV1或者SENetV2的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148439170)  |
| [yolov11改进系列]基于yolov11使用GhostNetV1或者GhostNetV2替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148438519)  |
| [yolov11改进系列]基于yolov11使用EfficientNetV1或者EfficientNetV2替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148425499)  |
| [yolov11改进系列]基于yolov11引入通道混洗的重参数化卷积RCS-OSA的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148421892)  |
| [yolov11改进系列]基于yolov11引入轻量级通用上采样算子CARAFE的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148419949)  |
| [yolov11改进系列]基于yolov11引入轻量高效动态上采样算子DySample提升小目标检测能力的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148414605)  |
| [yolov11改进系列]基于yolov11引入深度可分卷积与多尺度卷积的结合MSCB的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148413994)  |
| [yolov11改进系列]基于yolov11使用FasterNet替换backbone用于轻量化网络的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148409430)  |
| [yolov11改进系列]基于yolov11引入空间通道系统注意力机制SCSA的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148404219)  |
| [yolov11改进系列]基于yolov11引入自集成注意力机制SEAM解决遮挡问题的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148401428)  |
| [yolov11改进系列]基于yolov11使用图像去雾网络UnfogNet替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148400393)  |
| [yolov11改进系列]基于yolov11使用大卷积核UniRepLKNet替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148398882)  |
| [yolov11改进系列]基于yolov11使用极简主义网络VanillaNet替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148386687)  |
| [yolov11改进系列]基于yolov11引入特征融合注意网络FFA-Net的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148386106)  |
| [yolov11改进系列]基于yolov11引入多尺度空洞注意力MSDA的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148363974)  |
| [yolov11改进系列]基于yolov11引入迭代注意力特征融合iAFF的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148359270)  |
| [yolov11改进系列]基于yolov11引入上下文锚点注意力CAA的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148359007)  |
| [yolov11改进系列]基于yolov11引入可变形注意力DAttention的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148358890)  |
| [yolov11改进系列]基于yolov11引入跨空间学习的高效多尺度注意力EMA的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148352598)  |
| [yolov11改进系列]基于yolov11引入重参数化模块DiverseBranchBlock的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148349249)  |
| [yolov11改进系列]基于yolov11引入高效上采样卷积块EUCB的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148348793)  |
| [yolov11改进系列]基于yolov11引入高效坐标注意力机制CoordAttention的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148348371)  |
| [yolov11改进系列]基于yolov11引入全局注意力机制GAM的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148343985)  |
| [yolov11改进系列]基于yolov11引入轻量级注意力机制模块ECA的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148334006)  |
| [yolov11改进系列]基于yolov11引入双通道注意力机制CBAM的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148333272)  |
| [yolov11改进系列]基于yolov11引入大型分离卷积注意力模块LSKA减少计算复杂性和内存的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148322154)  |
| [yolov11改进系列]基于yolov11引入倒置残差块块注意力机制iEMA的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148304867)  |
| [yolov11改进系列]基于yolov11引入反向残留移动块注意力机制iRMB的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148296738)  |
| [yolov11改进系列]使用ConvNeXtV2替换backbone用于增强特征学习和多样性的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148295474)  |
| [yolov11改进系列]基于yolov11引入双卷积DualConv用于轻量化网络的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148295325)  |
| [yolov11改进系列]基于yolov11引入高效卷积模块SCConv减少冗余计算并提升特征学习的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148286016)  |
| [yolov11改进系列]基于yolov11引入混合标准卷积与深度可分离卷积GSConv用于轻量化网络的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148282956)  |
| [yolov11改进系列]基于yolov11引入空间深度转换卷积SPDConv用于低分辨率图像和小物体的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148280001)  |
| [yolov11改进系列]基于yolov11引入动态卷积DynamicConv的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148275316)  |
| [yolov11改进系列]使用轻量级反向残差块网络EMO替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148268939)  |
| [yolov11改进系列]基于yolov11引入可改变核卷积AKConv的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148261874)  |
| [yolov11改进系列]基于yolov11引入分布移位卷积DSConv的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148256082)  |
| [yolov11改进系列]基于yolov11引入全维度动态卷积ODConv的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148255417)  |
| [yolov11改进系列]基于yolov11引入异构卷积HetConv提升效率而损失准确度的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148243881)  |
| [yolov11改进系列]基于yolov11引入感受野注意力卷积RFAConv的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148243514)  |
| [yolov11改进系列]基于yolov11引入级联群体注意力机制CGAttention的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148227705)  |
| [yolov11改进系列]基于yolov11引入可切换空洞卷积SAConv模块python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148222713)  |
| [yolov11改进系列]基于yolov11引入轻量级Triplet Attention三重注意力模块python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148220532)  |
| [yolov11改进系列]基于yolov11轻量化下采样操作ADown改进Conv卷积减少参数量python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148218211)  |
| [yolov11改进系列]基于yolov11引入特征增强注意力机制ADNet的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148217834)  |
| [yolov11改进系列]基于yolov11引入自注意力与卷积混合模块ACmix提高FPS+检测效率python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148217649)  |
| [yolov11改进系列]基于yolov11的修改检测头为自适应特征融合模块为ASFFHead检测头的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148217483)  |
| [yolov11改进系列]基于yolov11的骨干轻量化更换backbone为shufflenetv1网络python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148213613)  |
| [yolov11改进系列]基于yolov11的骨干轻量化更换backbone为shufflenetv2网络python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148204899)  |
| [yolov11改进系列]基于yolov11引入双层路由注意力机制Biformer解决小目标遮挡等问题python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148202150)  |
| [yolov11改进系列]使用轻量级骨干网络MobileNetV1替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148200268)  |
| [yolov11改进系列]使用轻量级骨干网络MobileNetV2替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148199706)  |
| [yolov11改进系列]使用轻量级骨干网络MobileNetV3替换backbone的python源码+训练源码 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148195724)  |
| [yolov11改进系列]使用轻量级骨干网络MobileNetV4替换backbone的python源码+训练源码+改进流程+改进原理 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148193577)  |
| [yolov11改进系列]基于yolov11引入混合局部通道注意力机制MLCA的python源码+训练源码+改进原理+改进流程 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148190800)  |
| [yolov11改进系列]基于yolov11添加SE注意力机制python源码+训练源码+改进原理+改进流程 |  [详情](https://blog.csdn.net/FL1623863129/article/details/148184883)  |