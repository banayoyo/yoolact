# YOLACT: Real-time Instance Segmentation

> Author: Daniel Bolya, Chong Zhou, Fanyi Xiao, Yong Jae Lee (Thx for their work.)
>
> Year: 2019

## Abstract

 模型：简单的全卷积神经网络

模型效果：29.8mAP on MS COCO at 33fps  on a single Titan Xp，总而言之就是速度快于现有的任何一种方法。训练过程仅仅需要一块GPU，不需要搞分布式训练，轻量！实时的最低要求是30fps。

模型分割子任务：

1. 生成原型掩模mask集合。
2. 预测掩模mask的回归系数。
3. 通过原型掩模和回归系数生成实例掩模mask。

因为整个整个过程不依赖于 repooling，生成的mask质量高，且时序稳定。

此外论文还分析了模型在紧急行为下，可以以一种变化的翻译方式，自动学习定位实例。最后提出了一种更快的FMS，比标准的NMS少了12ms。（注：NMS 非极大值抑制，Non-maximum suppression）

## Introduction

> “Boxes are stupid anyway though, I’m probably a true
> believer in masks except I can’t get YOLO to learn them.”
> 							– Joseph Redmon, YOLOv3

论文开头引入了近年来的一些state of the art方法，如Mask RCNN，FCIS，这些模型是在高精度物体检测模型如Faster R-CNN，R-FCN的基础上发展而来的，但是过于追求模型performance而忽视了speed。本文作者要做的工作就是像SSD，YOLO这些一阶段的实时检测模型一样，搞一个一阶段的实例分割模型，算是填补实时实例分割方面的空缺，所以就有了YOLACT。

但是实时实例分割也不是一件容易的事情，比实时物体检测要困难。实时物体检测，是基于二阶段检测的方法，对第二阶段进行了移除，并且利用了某种方法来减少其损失的性能。但对于实时实例分割来说就不好以同样的方式进行扩展，因为现有的两阶段实时实例分割模型严重依赖于特征定位来产生mask，过程是先repooling一些box region 如ROI等等，然后根据这些定位的特征预测其掩模，具有串行的特点，难以加速。而现有的单阶段模型如FCIS虽然能够并行处理repooling和prediction，但是却需要在定位之后进行大量并且重要的后处理，无法做到实时效果。

为了解决上述的问题，本文提出了YOLACT，一种实时实例分割的模型框架，该框架放弃了显式的定位步骤，而采用了一种并行的双子任务策略：一个是对整张生成一个非局部的mask字典；一个是对每一个实例预测一组线性回归系数，然后再使用线性组合合成一个最终的mask。简单来说，就是使用相应的预测系数对原型进行线性组合，然后使用预测边框进行裁剪。这种方法的优点是快速，可并行，生成高质量的mask，具备通用性的idea。

分割的任务主要是解决where和what的问题，线性回归系数和对应的检测分支是解决what，而生成mask是解决what。在消除重复框方面，本文还提出了一种新型的非极大值抑制方法，比标准的方法快了12ms。

## Related Work

* 两阶段实例分割，Mask-RCNN：第一阶段生成ROI（存在repooling），第二阶段分类并分割ROI。
* 一阶段实例分割，生成位置敏感图，同时组合mask位置敏感池化技术，或结合语义分割回归和方向预测回归。缺点是：虽快于二阶段但是仍需要repooling并且许多非平凡计算，如mask voting。
* 其他传统语义分割的方法：如边界检测，像素聚类，实例mask嵌入。这些操作存在多阶段，甚至是昂贵的聚类过程。
* 现有的实时实例分割，如straight to shapes 虽然快，可实时，但是精度大打折扣。Box2Pix依靠轻量的主干检测器GoogLeNetv1 and SSD以及手动调参的算法在数据集Cityscapes and KITTI实现了实时，但是却没有在COCO数据集上进行测试。而Mask R-CNN则是保持了COCO数据集上实例分割的最快速度，但是13.5fps仍显不足。
* 原型prototypes，常用在经典的计算机视觉中的特征表示。在本文中使用原型来进行组合mask。而且是针对特定的图像都能学习到对应的prototypes，而不是全局共享同一prototypes。

## YOLACT

YOLACT采用双并行子任务后集成生成mask策略。

1. 第一个分支使用FCN生成尺寸固定的原型mask集合，这些mask仍不属于任何一个实例。
2. 第二个分支为对象检测分支，在对象检测分支的开头，预测多个mask的回归系数向量，然后通过NMS减少相似的mask。向量的个数决定了最后的实例个数。这些单一的向量用于第一个分支生成的mask，通过线性组合的方式编码出新的实例。
3. 然后实例进行裁剪和阈值分割，最终生成分割结果。

基本原理

掩模的空间连贯性，而卷积层可以利用该连贯性，而全连接层则不行。而现有的单阶段检测器是通过全连接层为每一个anchor产生一个代表类别和box系数的输出。两阶段方法则是使用定位步骤ROI，虽然能保证mask的空间连贯性，但是却依赖第一阶段的RPN提供定位坐标，严重影响了速度。故我们将两阶段并行化，擅长生成语义向量的fc和擅长产生空间掩模的conv的两个过程同时进行，前者产生mask回归系数，后者产生原型掩模。独立计算两部分，主干网络的计算开销主要来自组装步骤，这可以通过简单的矩阵乘法来实现，这样我们就既维持了特征空间的连贯性同时又保证是快速的一阶段模型。

个人理解，这里所谓的Prototype就是某个卷积层的特征图，可以一定的语义表征作用。以下是架构图。
![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190430172822.png)

### Prototype Generation

对于原型生成的分支，作者称之为protonet，专用于对某一张图片产生prototype集合，该网络使用FCN实现，最后输出k个通道，即k个特征图，每一个特征图就代表一个原型。这种方法近似于标准的语义分割范式，但是不同之处在于没有对prototype进行损失函数的计算，监督过程是在最后mask集成之后进行。
![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190430173437.png)

作者提出了两种重要的设计选择，一个是从深度的骨干特征中生成更具有鲁棒性的mask，而高分辨率的prototype则能生成质量更高的mask，以及在小物体上也能有更好的表现。所以使用FPN的方法中最大的特征层P3作为protonet的输入，这样既保证了较强的鲁棒性，同时又保证了高分辨率。而protonet的输出是无边界的，因为这允许网络以更大的，压倒性的激活机制来产生置信度更高的prototype。这里选择了RELU作为激活函数。

### Mask Coefficients

典型的物体检测方式是通过分类分支和预测边框分支进行的。而掩模回归系数预测，仅仅需要提供第三个并行分支来预测k个掩膜系数，每一个对应一个prototype。也就是说对于每一个anchor，我们产生的系数有 4 + c + k。即边界系数，类别系数，和掩模系数，分别用于定位，分类，对原型回归得到实例分割图。

在最后的mask系数输出中，作者们发现很有必要减去原来的prototype，使用了tanh作为k个掩模系数的激活函数，保证产生非线性且稳定的输出。很显然如果在线性组合中若不存在为负的系数，那样的话是无法构造实例mask的。

![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190504214945.png)

### Mask Assembly

回归系数和prototype集合进行线性组合，得到一个组合的mask，最终需要输入sigmoid函数映射到【0，1】，整个操作可以简单的使用矩阵乘法和sigmoid函数实现。

![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190430195447.png)

P 是一个 h * w * k的原型掩模集合，C为 n * k的掩模系数矩阵，对应了 n 个实例的回归mask系数向量，这些系数向量经过了非极大值抑制和阈值处理。为了简单快速起见，这里仅仅使用了基本的线性回归模型。

损失函数设计：

* 分类误差：同SSD
* box回归误差：同SSD
* mask误差：BCE loss 

mask 裁剪：为了保留原型中的小物体，我们在评价过程中使用预测的边框对最后集成的mask进行了裁剪。在实际训练中，却使用了ground truth bounding box，对Lmask除以true box area。

### Emergent Behavior

常识认为FCN是需要平移不变性的，并且需要添加平移方差在任务中。如FCIS mask R-CNN尝试添加平移方差，不管是在方向映射步骤还是位置敏感的repooling，或者是把mask分支放在第二阶段。

在本文的方法中，唯一需要添加平移方差的地方则是使用预测方框对final mask进行裁剪的时候，但是即便不添加这一步在实际的项目中，也能work。YOLACT学习如何在原型集合上经过不同的激活来进行实例的定位。

### Backbone Detector

ResNet101+FPN提取特征，默认图像尺寸为550 * 550。

* 修改FPN中的一些层

* focal loss

* 分类和框回归采用SSD的 loss设计

### Other Improvements

Fast NMS：并行消除重复边框，已经消除的框能够对现有的其他框产生抑制效果。但也是速度换性能代价。

语义分割损失：训练时增加格外损失计算，但预测却取消该模块计算。

## Results

使用数据集：MS COCO‘s 实例分割任务数据集

度量方式：FPS AP

模型：YOLACT-550   backbone：R-101-FPN。

State of the art methods：Mask R-CNN 、FCIS 、PA-Net、 RetinaMask 、MS R-CNN

从结果上来看，YOLACT-500速度远胜其他baseline models，FPS超过了30，是目前最快方法的3.8x倍，而精度方面AP值却低于30，逊于Mask R-CNN、MS R-CNN等，仅优于FCIS。

## Discussion

YOLACT算法目前存在着两大常见典型错误，如下：

1. 定位失败。当一个场景某一个地方中存在较多的物体时，网络定位物体的能力较弱，容易出现边框不精准的情况，此时实例分割的结果不完整。
2. 没有抑制裁剪引入的噪声。当定位框不精确时，引入了实例以外的噪声。

目前可选的解决方案有：a mask error down-weighting scheme like in MS R-CNN。

个人认为可以添加focal loss来帮助改善mask的误差，有待验证。