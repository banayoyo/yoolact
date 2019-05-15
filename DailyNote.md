# 日常笔记

本文件主要记录个人在复现该论文过程中的一些笔记,以todo和随笔的形式进行。

## 一、论文阅读

- [x] 完成摘要和引言解读。

记：简单来说，模型就是简单又快，基于并行的双子任务完成定位和分类。

- [x] 初步完成模型架构解读

记：YOLACT实例分割的思路简单来说，其步骤就是：

1. 借鉴了Resnet101 + FPN 模型来提取图像特征，做图像特征预提取；
2. 通过两个并行的分支处理特征得到初步的实例分割结果：
   1. 一个分支protinet提取图像的多个原型mask集合；
   2. 另外一个分支：
      1. 改进Retinanet结构，通过三个子分支回归得到mask系数向量、分类结果，以及边框相关值。（借鉴SSD损失函数）
      2. 经过快速的NMS得到最终分类定位结果  
3. 通过mask回归系数和原型mask集合进行线性组合得到final mask
4. 使用分类后的边框对final mask进行进一步的裁剪，分离不同实例
5. 进行阈值处理，提取干净的mask；
6. mask，边框位置，分类结果具备了，呈现实例分割效果。

- [x] 完成模型结果解读和讨论

记：快约3.8倍，但精度降低1/5左右。

更多解读内容请看[PaperNote](https://github.com/JoshuaQYH/yolact-pytorch/blob/master/PaperNote.md)。

## 二、代码阅读

由于yolact模型是在现有的几个模型的基础上发展而来的，里头某些部分如FPN，FCN，SSD,Retinanet等等来自前人的论文成果，必须对这些内容有一定的了解和认识才能真正理解yolact模型的运行机制。以下从代码入手，并贴上相应的博客辅助理解。

### 优先

- [x] [FPN](https://github.com/jwyang/fpn.pytorch)--FPN提取特征是yolact的主要方法。
- [x] [SSD](https://github.com/amdegroot/ssd.pytorch)--yolact借鉴了SSD的损失函数。
- [x] [FCN](https://github.com/wkentaro/pytorch-fcn) -- protonet分支使用了类似fcn的操作。
- [x] [yolact](https://github.com/dbolya/yolact)--yolact官方实现,官方代码较为全面庞杂，难以上手。

记：yolact官方实现的代码真的看的有点脑壳疼，有很多论文中没有提及的地方，尤其是其中各种参数的配置，看的有点乱，
可能是和我平时的编码风格过于不同，导致我看的有点费劲；另一方面就是yolact代码需要经过大量的模型架构和超参实验，作者的的代码方法扩展性比较强（牺牲了一定的易读性），但是对于我们来说，从学习的角度来讲，就不用太过要求扩展性了。
本次复现，在方法和网络结构方面，尽可能贴近论文论述内容。

### 可选

- [ ] [Detectron-RetinaNet](https://github.com/facebookresearch/Detectron)--yolact借鉴了该单阶段检测模型，有必要了解。
- [ ] [Faster R-CNN](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)--学习两阶段检测模型中RPN子网如何生成候选区域。
- [ ] [Mask R-CNN](https://github.com/facebookresearch/maskrcnn-benchmark)--学习经典实例分割方法，帮助理解对比。
- [ ] [Mask Scoring R-CNN](https://github.com/zjhuang22/maskscoring_rcnn)--前沿优秀的实例分割方法，试图借鉴优化。

推荐：

- [SSD pytorch实现教程](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)--看完醍醐灌顶！课程TA也是借鉴的这个，赞！

## 三、COCO数据集探索

- [COCO数据集详解](https://blog.csdn.net/wc781708249/article/details/79603522)
- [Microsoft COCO 数据集解释](https://blog.csdn.net/u012905422/article/details/52372755)
- [COCO数据集官网](http://cocodataset.org/#download)--需要翻墙
- [COCO数据集官方读取加载API](https://github.com/cocodataset/cocoapi)
- [COCO数据集的API使用](https://blog.csdn.net/qq_41847324/article/details/86224628)
- [COCO数据集的下载方法](https://blog.csdn.net/qq_33000225/article/details/78831102)--不需要翻墙就可以下载2014版本数据集。

由于设备、资源和时间问题，在这里仅仅考虑下载2014版本数据集，其中有20G左右的图片和500M左右的标签文件。
该数据集主要有的特点如下：

1. 每张图片有多个目标
2. 可用于目标检测，情景识别
3. 超过300,000张图片
4. 超过2000,000个实例
5. 80种目标
6. 每张图片有5个captions
7. 在10,000个人上进行关键点的标注

最终的下载解决方案是使用wget的方法直接获取官网数据，写了下载脚本,见`\data\download.sh`，奇怪的是我没有翻墙上不来官网，但是wget却能访问到，真是神奇。起初用了官方yolact下载方法，使用的curl，一开始速度还可以，后来速度太慢了，要下一天多，后来请教了CSDN博主，转用了wget，速度还挺快的，不到6个小时就下载了20G文件。

## 代码初步实现

### model

- [x] backbone resnet101(其实是原fpn的左半部分，从中抽取某些层作为以下fpn的输入）
- [x] fpn(其实是原fcn的右半部分）
- [x] protonet (fcn)
- [x] prediction branch (类似retinanet)
- [x] yolact net 组合以上subnets。
