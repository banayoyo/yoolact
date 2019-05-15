"""
Author: JoshuaQYH

Description:
    本文件是用于构建yolact网络模型，其模型结构简化如下。

    YOLACT model
    |____________1. Resnet backbone：作为模型的第一部分网络，提取图像特征，这里的Renet充当FPN的左半部分
    |
    |____________2. FPN: 作为模型的第二部分网络，使用特征金字塔进行图像特征的增强，这里的FPN其实只是右半部分，和resnet特定层add
    |
    |____________3.1 prediction branch: 
    |                |___ 3.1.1: bbox 预测分支,仿retinanet，损失函数仿ssd
    |                |___ 3.1.2: class 预测分支，仿retinanet，损失函数仿ssd
    |                |___ 3.1.3: mask prototype coefficients 预测分支
    |                |___ NMS处理，减少重复
    |
    |____________3.2 protonet branch: 使用FCN生成待线性组合的mask prototype
    |
    |____________4. ensembel: 集成 3.1 3.2的结果，线性组合得到mask，计算损失函数。 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
import config as cfg

class BottleNeck(nn.Module):
    """
    使用resnet的第二种基本块，bottleneck使用三层版本，结构如下：
    输入的特征图有256个channel，一共有三个卷积层。
    x -> conv 1×1（256,64） -> bn + relu ->  conv 3×3（64, 64) -> bn + relu -> conv 1×1(64, 256) -> y + x -> relu ->output
    |__________________________________________________________________________________________________↑
    代码借鉴了 torchvision.model.resnet https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
    """
    expansion = 4  #  输入特征图通道数的扩张倍数, in * 4 = out

    def __init__(self, inplanes, planes, downsample=None, dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleNeck, self).__init__()
        """
        @ params:
            * inplanes: 输入特征图的通道数

            * planes: 中间卷积生成的特征图的通道数

            * downsample: 在中间卷积过程中可能发生特征图尺寸缩小或者与输入通道不一致，需要对原来输入进行下采样或改变通道数

            * dilation: 膨胀卷积的尺度，膨胀卷积可以在进行卷积时候，增大感受野，但是不缩小特征图的尺寸

            * norm_layer: 权重标准化层，默认 bn。

        """
        self.conv1 = nn.Conv2d(inplanes, planes, stride=1, dilation=dilation,bias=False,kernel_size=1, padding=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride=1, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False, dilation=dilation, padding=1)
        self.bn3 = norm_layer(planes * self.expansion)

        self.down_sample = downsample

    def forward(self, x):
        residual = x 

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.down_sample is not None:
            residual = self.donw_sample(residual) 

        out = self.relu(out + residual)
        
        return out

class ResNetBackBone(nn.Module):
    """
        resnet 架构设计参考了 pytorchvision model 和 yolact 官方实现版本
        yolact： https://github.com/dbolya/yolact/blob/master/yolact.py
        pytorchvision resnet：https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
    """

    def __init__(self,layers,block=BottleNeck, 
                norm_layer=nn.BatchNorm2d,zero_init_residual=False):
        """
        @ params:
            * layers: 以列表的方式指定每层生成的bottleneck个数

            * block: resnet的基本块

            * norm_layer: 权重标准化层

            * zero_init_residual: 决定是否对bottleneck中的bn层进行常量初始化,来自 pytorchvision resnet model

        """
        super(ResNetBackBone, self).__init__()
        
        # 多个bottleneck构成一个 baselayer，由函数make_layer生成
        self.num_base_layer = len(layers) 
        # 用于存放resnet的网络模块,存放每一个每一次make_layer的结果
        # self.layers = nn.ModuleList() 
        # 存放每一个baselayer的输出通道数
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1

        self.inplanes = 64

        # 第一层卷积，输入为3通道的图像
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        assert(len(layers) == 4)
        # 生成多个bottle neck list，作为base layer,make的结果append到了self.layers列表中
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 以下权重初始化来自torchvision resnet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
            这里所make的layer是包含了串行的多个block
        """

        # 构造block中的下采样层
        downsample = None
        if stride != 2 or self.inplanes != planes * block.expansion:
            # 使用 stride > 1来进行下采样,卷积核为 1 × 1
            downsample = nn.Sequential(*[nn.Conv2d(self.inplanes, planes * block.expansion,
                                        kernel_size=1, stride=stride, dilation=self.dilation),
                                        self.norm_layer(planes * block.expansion)])

        layers = []
        layers.append(block(self.inplanes, planes, downsample=downsample,dilation=self.dilation,norm_layer=self.norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        self.channels.append(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        inputs = self.maxpool(inputs)
        x1 = self.layer1(inputs)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1,x2,x3,x4


class FPN(nn.Module):
    """
        网络中的FPN特征金字塔提取特征部分。
        实现了FPN的通用版本，论文地址： https://arxiv.org/pdf/1612.03144.pdf

        FPN的大概过程就是：
        1. 先下采样卷积，得到多个层级的尺寸逐渐缩小的特征图，构成金字塔 A
        2. 顶层最小的特征图可采用双线性插值的方式上采样，扩大尺寸，得到 a
        3. 次顶层的特征图经过横向1*1卷积进行降维，得到 b
        4. a + b 然后再进行卷积，得到新的特征图。
        5. 新的特征图代替 2 中的特征图，重复步骤 2 ～ 4若干次，构成金字塔 B

        ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        ！！！ 注意，从官方实现版本的FPN来看，它形成金字塔A的过程是通过backbone来实现的，如 Resnet101，
        ！！！ 然后从backbone其中抽取若干层输出的特征图作为金字塔A的特征图层，所以在这里没有金字塔A的构建过程                         
        ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    """
    def __init__(self, input_channels, num_features, num_down_layers):
        """
        初始化特征金字塔，其实只有原来版本的右边部分金字塔 B，左边部分金字塔 A 被backbone部分不同尺寸的特征层输出所替代了
        @ params:
            * input_channels: 横向卷积的输入通道数目列表，其元素表示来自金字塔A的某一层的特征图通道数。排序从金字塔底层开始
            * num_features: 每一层卷积输出的特征图数目
            * num_down_layers: 降采样的层数
        """
        super(FPN, self).__init__()
        self.num_features = num_features
        self.num_down_layers = num_down_layers
        # 以下初始化三种类型的卷积层

        # 从金字塔顶层开始往下生成多个横向的卷积核为1 × 1的卷积层，用于降维,对应论文 fig2中 O+ 符号,
        self.lat_layers = nn.ModuleList([nn.Conv2d(x, num_features, kernel_size =1) \
                                        for x in reversed(input_channels)]) 

        # 自顶向下初始化对上采样后相加的特征图的卷积层，用于提取特征，个数同横向卷积层,对应论文fig2 生成 P4,P3的步骤
        self.pred_layers = nn.ModuleList([nn.Conv2d(num_features, num_features, \
                                        kernel_size=3, padding=1) for _ in input_channels])

        # 降采样层, 对应论文中 fig2中生成 P6，P7的步骤,使用conv的 stride=2来进行下采样
        self.downsample_layers = nn.ModuleList([nn.Conv2d(num_features, num_features,kernel_size=3, \
                                              padding=1, stride=2) for _ in range(num_down_layers)])

    def _upsample_add(self, x, y):
        """
        对 x 特征图进行上采样，然后和 y 特征图相加。
        """
        h,w = x.size(2), x.size(3)
        return F.upsample(x, size = (h, w), mode='bilinear') + y

    def forward(self, convouts):
        """
        从backbone中抽取若干层的特征图输出，准备和上采样后的特征图进行相加。
        @params:
            convouts：FPN左半边金字塔 A 的特征图输出列表，该列表中的特征图层数对应这input_channels中的元素, 特征图从大到小，自底向上
        @return:
            out: 金字塔各层的特征图，排序从金字塔底层开始，特征图由大到小。
        代码参考了官方实现版本 https://github.com/dbolya/yolact/blob/master/yolact.py 此处对其进行了简化
        """
        assert len(convouts) == 3
        # 存放金字塔各层的特征图,自底向上排列
    

        C3,C4,C5 = convouts

        

        # 上采样，得到多个特征图层，fig2中 由 P5产生 P4，P4产生P3的
        for i in range(len(convouts)):
            j = - i - 1
            if i > 0:
                x = self._upsample_add(x, self.lat_layers[i](convouts[j]))
            elif i == 0:
                x = self.lat_layers[i](convouts[j])
            out[j] = F.relu(self.pred_layers[i](x))
        
        # 向下采样，得到多个特征图层。
        for idx in range(self.num_down_layers):
            out.append(self.downsample_layers[idx](out[-1]))
        return out

class PredictionBranch(nn.Module):
    """
        网络中的预测分支，也就是论文中所说的 Head Architecture。
        最终有三个小分支用于边框回归，预测类别，以及预测prototype mask的回归系数。
    """
    def __init__(self, num_classes, input_channels, mid_channels=256, 
                 aspect_ratio=[1], scales=[1], num_mask_coeffs=64):
        """
            * num_classes: 预测的类别总数

            * input_channels: 输入特征图的通道数(来自FPN

            * mid_channels: Head Architecture 在分支之前的最后一个卷积层的输出通道数

            * aspect_ratio: 先验框的宽高比例列表，在不同scales下也有不同的列表，所以是二维的。
                            每一维的list代表该scale下的比例列表。

            * scales: 表示当前卷积层先验框的缩放比例。此处借用官方代码的例子来说明一下：
                      如果原图尺寸是 600×600px，而当前卷积层尺寸是30 × 30px，
                      那么当前卷积层的每一个px相当于了原图的20×20px，一个先验框如果占据了当前卷积层2px的空间，
                      如果 scale = 2，那么当前先验框映射到原图的大小就是 20 × 20 × 2 × 2。scale就是先验框的缩放比例系数。

            * num_mask_coeffs: 网络输出的用于mask prototype线性回归的系数个数，等于protonet最后输出的prototype的层数。官方参数是64。
        """
        super(PredictionBranch, self).__init__()

        self.num_classes     = num_classes
        self.aspect_ratio    = aspect_ratio
        self.scales          = scales
        self.num_mask_coeffs = num_mask_coeffs
        self.num_prior_boxes     = len(aspect_ratio) # 每个位置先验框生成的个数

        self.prior_boxes = None
        self.last_conv_size = None

        """
        网络设计定义,完全按照论文中的Fig4

        conv1(input_channels, 256) -> Relu -> conv2(256, mid_channels) -> relu

        -->1: 预测类别分支：class_layer(mid_channels, num_classes * num_prior_boxes)

        -->2: 边框回归分支：bbox_layer(mid_channels, num_anchor * 4)

        -->3: mask回归系数分支：mask_coefs_layer(mid_channels, num_mask_coeffs) -> tanh()
        
        上述网络层的卷积核均为 3 × 3，stride = 1， padding = 1，保证feature map尺寸不变。

        """
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=256,kernel_size=3,stride=1,padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=256,out_channels=mid_channels,kernel_size=3,stride=1,padding=1)
        
        self.class_layer = nn.Conv2d(in_channels=mid_channels,out_channels=num_classes * self.num_prior_boxes,
                                    kernel_size=3,stride=1,padding=1)

        self.bbox_layer = nn.Conv2d(in_channels=mid_channels,out_channels=self.num_prior_boxes*4,
                                    kernel_size=3,stride=1,padding=1)

        self.mask_coefs_layer = nn.Conv2d(in_channels=mid_channels,out_channels=num_mask_coeffs * self.num_prior_boxes,
                                    kernel_size=3,stride=1,padding=1)

        self.mask_coefs_activation_func = nn.Tanh()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        @ param：
            x：FPN的特征输入,尺寸为 [batch_size, channels, weight, height]，简写为[B,C，H，W]
        
        @ return:
            * pred_bboxes:  [B, H * W * num_prior_boxes，4] 
            * pred_classes: [B, H * W * num_prior_boxes, num_classes]
            * pred_coeffs:  [B, H * W * num_prior_boxes, num_mask_coeffs] 
            * prior_bboxes: [H * W * num_prior_boxes, 4]  不同尺度的输入特征图产生的先验框数量不同
        """

        # 获取输入特征图的长宽
        conv_h = x.size(2)
        conv_w = x.size(3)

        # 计算先验框集合
        prior_bboxes = self.make_prior_bboxes(conv_h, conv_w)
        
        # 简单卷积层预处理
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        """
        三个小分支的卷积运算
        permute 的作用是起到维度转换的效果,原来是[B,C,H,W]经过(0,2,3,1)置换之后，变为[B,H,W,C]
        contiguous 的作用是对置换的结果进行一次拷贝，便于后续的view，如 B 是 A 的拷贝，那么修改 B，对A不会有影响。
        view(B, -1, x) 中 -1 的作用是当指定了0，2维的尺寸(B，x)时，可自动推断出第 1 维的尺寸。
        """
        pred_bboxes = self.bbox_layer(out).permute(0,2,3,1).contiguous().view(x.size(0), -1, 4)
        pred_classes = self.class_layer(out).permute(0,2,3,1).contiguous().view(x.size(0), -1, self.num_classes)
        pred_coeffs = self.mask_coefs_layer(out).permute(0,2,3,1).contiguous().view(x.size(0), -1, self.num_mask_coeffs * self.num_prior_boxes)
        pred_coeffs = self.mask_coefs_activation_func(pred_coeffs)

        p_out = {'loc': pred_bboxes, 'conf': pred_classes, 'coeffs': pred_coeffs, 'prior': prior_bboxes}
        return p_out

    def make_prior_bboxes(self, conv_h, conv_w): 
        """
        给定一张特征图的尺寸，仿照SSD思想，计算返回该特征图上所有先验框。
        先验框的格式是 [cx, cy, width, height]， 
        cx, cy为先验框的中心坐标,其坐标范围在(0,1]之间,特征图的比例坐标。
        width,height为特征图的比例尺寸，范围在(0,1]之间

        值得注意的是，这里的width，height的大小是图片比例，由于特征图比例没有发生变化，和 gt box 运算时，
        gt box的width，height同样需要变为对应的图片比例，这样的话，不管是什么尺寸的prior box，其 scale 都是 1。
   
        """

        # 如果当前特征图跟上一次计算的特征图尺寸一样时，可直接返回上一次的先验框结果，减少重复计算
        if self.last_conv_size != (conv_h, conv_w):
            prior_boxes = []
            for j in range(conv_h):
                for i in range(conv_w):
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h

                    # scale为元素，ars为一维列表
                    for scale, ar in zip(self.scales, self.aspect_ratio):
                        w =  scale * ar / conv_w # ！！！
                        h = scale / ar / conv_h
                        prior_boxes.append([x, y, w, h])
            self.prior_boxes = torch.FloatTensor(prior_boxes).view(-1, 4).clamp_(0,1)
            self.last_conv_size = (conv_h, conv_w)
        return self.prior_boxes
        

class ProtoNetBranch(nn.Module):
    """
        按照fig3的 protonet 架构图进行复现
    """
    def __init__(self, input_channels, num_mask_protos, deconv_up_sample=False):
        """
        @ params:
            * input_channels: 网络输入的特征图通道数
            * num_mask_protos: 网络输出的特征图通道数,用于与predict分支的coeff系数做线性回归
        
        模型结构,使用全卷积模型，中间使用双线性插值or反卷积进行上采样,
        3*3 conv1 2 3 -> upsample(bilinear) / deconv -> 3*3 conv 4 -> 1*1 conv 5
        """
        super(ProtoNetBranch, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        
        # 反卷积进行上采样，尺寸扩大一倍
        # 反卷积尺寸变换公式见 https://pytorch.org/docs/stable/nn.html?highlight=convtran#torch.nn.ConvTranspose2d
        # 尺寸变为原来的两倍有多种方式，以下是其中一种（按照公式进行推导）
        self.deconv = deconv_up_sample
        if self.deconv: 
            self.updeconv = nn.ConvTranspose2d(in_channels=256,out_channels=256, kernel_size=3,stride=2,output_padding=1)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, input_channels, kernel_size=1,stride=1)

        self.relu = nn.ReLU(inplace=True)

    def up_sample_layer(self, x):
        # yolact 官方实现版本 采样使用双线性插值+普通卷积 
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))

        if self.deconv:
            out = self.updeconv(out)
        else:
            out = self.up_sample_layer(out)
        
        out = self.relu(self.con4(out))
        out = self.relu(self.conv5(out))
        return out


class YOlACT(nn.Module):

    def __init__(self, backbone_type):
        super(YOlACT, self).__init__()
        
        # 定义 backbone 提取特征
        if backbone_type == "resnet101":
            self.backbone = ResNetBackBone(cfg.resnet101['layers'])
            self.selected_layers = cfg.resnet101['selected_layers']
        
        src_channels = self.backbone.channels

        # 定义fpn，再次提取特征
        # 选择backbone中的某些层的输出与fpn融合
        self.fpn = FPN([src_channels[i] for i in self.selected_layers], 
                        cfg.fpn['num_features'], cfg.fpn['num_downsample'])

        # 定义预测网络模块。fpn的每一层特征图输出（对应论文中的P5，P4，P3），都作为预测分支的输入。
        self.prediction_layre = PredictionBranch(cfg.num_classes, src_channels[0], src_channels[0],
                                    aspect_ratio=cfg.pred_setting['pred_aspect_ratio'][0],
                                    scales=cfg.pred_setting['pred_scales'][0], num_mask_coeffs=cfg.num_mask_coeff)

        
        # 定义 protonet, 使用特征图最大的一层的通道数，对应论文中的 P3
        self.proto_net = ProtoNetBranch(src_channels[0], num_mask_protos=cfg.num_mask_proto)

        # 在评估时使用
        self.detect = None

        self.freeze_bn()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

        
    def forward(self, x):
        
        x1,x2,x3,x4 = self.backbone(x)

        outs = self.fpn([x2, x3, x4])

        proto_x = outs[0] # fpn 最大的特征图层作为 protonet的输入
        proto_out = self.proto_net(proto_x)
        
        proto_map = proto_out.clone()  # 拷贝protonet输出的特征层

        # 便于后续进行矩阵乘法
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        # 预测分支的输出
        pred_outs = {'loc':[], 'conf':[], 'coeffs':[], 'priors':[], 'proto': None}

        for idx, pred_layer in zip(self.selected_layers, self.prediction_layres):
            pred_x = outs[idx]
            p_out = pred_layer(pred_x)
            for k, v in p_out.items():
                pred_outs[k].append(v)
        
        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)
        
        pred_outs['proto'] = proto_out

        if self.training:
            return pred_outs
        elif self.eval:
            pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
            # return self.detect(pred_outs)
