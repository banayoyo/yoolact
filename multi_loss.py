"""
Author: JoshuaQYH
Desciption:
    定义整个模型损失函数，借鉴了SSD模型
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as Variable
import config as cfg 
import utils 
from torch.autograd import Variable

class MultiLoss(nn.Module):
    """
    SSD 损失函数
    """
    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio, l1_alpha):
        super(MultiLoss, self).__init__()

        self.num_classes = num_classes

        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio

        self.l1_alpha = 1

        self.smoth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

        self.prior_xy = None

    def forward(self, predictions, wrapper, wrapper_mask):
        # pred_outs = {'loc':[], 'conf':[], 'coeffs':[], 'priors':[], 'proto': None}
        """
            * loc:  [B, H * W * num_prior_boxes，4] 
            * conf: [B, H * W * num_prior_boxes, num_classes]
            * coeff:  [B, H * W * num_prior_boxes, num_mask_coeffs] 
            * priors: [H * W * num_prior_boxes, 4]  不同尺度的输入特征图产生的先验框数量不同
            * proto: [B, C, H, W], C 特征图通道数 等于 num_mask_coeffs
        """
        loc_data = predictions['loc']
        conf_data = predictions['conf']
        priors_data = predictions['priors']
        proto_data = predictions['proto'].permute(0,2,3,1).contiguous().view(loc_data.size(0), -1, cfg.num_mask_proto) # [B H*W C]
        coeff_data = predictions['coeff']

        if self.prior_xy is not None:
            self.prior_xy = utils.cxcy_to_xy(priors_data)
        
        """
            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.
        """
        targets, masks, num_crowds = wrapper.get_args(wrapper_mask)
        
        labels = [None] * len(targets)

        batch_size = loc_data.size(0)
        priors_data = priors_data[:loc_data.size(1),:]
        num_priors = priors_data.size(0)
        num_classes = self.num_classes

        # 初始化相同的尺寸的tensor,存储数据
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_box_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()

        defaults = priors_data.data

        assert num_priors == loc_data.size(1) == coeff_data.size(1) == conf_data.size(1)

        # 定位损失计算

        # 遍历每一张图像，匹配先验框和gt box
        for i in range(batch_size):
            #n_objects = targets[i].size(0)

            # 获取truths box坐标
            truths = targets[i][:, :-1].data
            # 获取box分类标签
            labels[i] = targets[i][:, -1].data.long()
            # 获取存在crowd的物体个数
            """
            cur_crowds = num_crowds[i]

            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes, truths = split(truths)

                # 不使用crowd 标签或者mask
                _, labels[idx] = split(labels[idx])
                _, masks[idx]  = split(masks[idx])
            else:
                crowd_boxes = None
            """

            # 由预测偏移值和先验框得到预测的框
            decoded_priors = utils.gcxgcy_to_cxcy(loc_data[i], self.prior_xy)

            # 计算每一张图片上每一个decode之后的预测框和gt box的 IOU值
            overlap = utils.find_intersection(truths, decoded_priors)

            # overlap 相当于一个二维矩阵，列名为 prior box，行名为 gt box。指定[行，列]可以得到对应prior box和gt box的IOU值。

            # 对每一个gtbox，找到其IOU最大的prior box,及其索引下标，按IOU值，降序排列
            _, best_prior_idx = overlap.max(1)

            # 对每一个prior box 找到其IOU最大的gt box，及其索引下标，按IOU值降序排列
            best_overlap_for_priors, best_overlap_idx = overlap.max(0)

            # 保证每一个先验框对应的IOU最大的gt box的IOU值为2，防止在后续的阈值处理中被过滤
            # best_overlap_for_priors.index_fill_(0, best_prior_idx, self.pos_threshold)

            # 创建index(prior, gt),每一个prior按索引排列，通过索引可以得到对应的IOU最大的先验框
            best_overlap_idx[best_prior_idx] = torch.LongTensor(range(best_prior_idx.size(0))).cuda("cuda:" + str(cfg.DEVICE_ID))

            # 确保每个最佳先验框匹配到的最佳gt box的阈值为 1
            best_overlap_for_priors[best_prior_idx] = 1.

            # 每一个prior匹配到最佳的的gt box坐标
            matches = truths[best_overlap_for_priors]  # [num_priors, 4]

            # 每一个prior匹配到的最佳gt box的类别，类别0为背景
            conf = labels[best_overlap_for_priors] + 1  # [num_priors, 1]

            # 使用双阈值进行过滤
            conf[best_overlap_for_priors < self.pos_threshold] = -1  # 中性样本,可能为背景或者前景，暂不区分
            conf[best_overlap_for_priors < self.neg_threshold] = 0   # 负样本，作为背景

            """
            if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
                crowd_overlaps = utils.find_jaccard_overlap(decoded_priors，crowd_boxes, iscrowd=True)
                best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
                # Set non-positives with crowd iou of over the threshold to be neutral.
                conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1
            """

            loc = utils.cxcy_to_gcxgcy(matches, self.prior_xy)
            loc_t[i] = loc
            conf_t[i] = conf  
            idx_t[i] = best_overlap_idx
            gt_box_t[i,:,:] = truths[idx_t[i]]

        loc_t  = Variable(loc_t, requires_grad=False)  # 每一个prior box与其最匹配gt box的偏移量
        conf_t = Variable(conf_t, requires_grad=False) # 每一个prior box与其最匹配的gt box的类别,整体加1
        idx_t = Variable(idx_t, requires_grad=False)   # 每一个prior box最匹配的gt box的索引值

        pos = conf_t > 0 # 获取prior box 类别为正的prior box位置。表示该box存在物体
        num_pos = pos.sum(dim=1,keepdim=True) # 求出每张图片上prior box存在物体的box的个数
        # 获取正样本的prior box的索引
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data) # [batch_size, num_priors, 4]

        losses = {}

        # 使用平滑l1损失函数计算边框回归损失    
        losses['box'] = self.smoth_l1(loc_data[pos_idx], loc_t[pos_idx], reduction='sum') * self.l1_alpha

        # 计算原型系数和原型特征图的线性组合损失
        losses['mask'] = self.lincomb_mask_loss(pos, idx_t, loc_data, priors_data, proto_data, masks, coeff_data, gt_box_t)
                
        # focal loss  计算分类的置信损失 
        losses['conf'] = self.focal_conf_sigmoid_loss(conf_data, conf_t)
        
        return losses

    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        搬运自官方 yolact
        """
        num_classes = conf_data.size(-1)

        conf_t = conf_t.view(-1) # [batch_size*num_priors]
        conf_data = conf_data.view(-1, num_classes) # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0 # can't mask with -1, so filter that out

        # Compute a one-hot embedding of conf_t
        # From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
        conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
        conf_pm_t  = conf_one_t * 2 - 1 # -1 if background, +1 if forground for specific class

        logpt = F.logsigmoid(conf_data * conf_pm_t) # note: 1 - sigmoid(x) = sigmoid(-x)
        pt    = logpt.exp()

        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (1 - conf_one_t)
        at[..., 0] = 0 # Set alpha for the background class to 0 because sigmoid focal loss doesn't use it

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(dim=-1)

        return cfg.conf_alpha * loss.sum()

    def lincomb_mask_loss(self, pos, idx_t, loc_data, priors, proto_data, masks, 
                          coeff_data, gt_box_t,interpolation_mode='bilinear'):
        """
        使用原型掩码系数和原型掩码进行线性组合，利用真实掩码计算BCE损失函数。
        @ params：
            * pos: 图片中存在物体的先验框索引  [batch_size, h*w*num_priors]
            * idx_t： 当前图片在这一批量图片的索引  
            * loc_data: 该批量图片的经网络预测的先验框偏移值  [batch_size, h*w]
            * priors： 该图片的先验框集合 [h * w * num_priors, 4]
            * proto_data: 原型掩码特征图 [batchs_size, h*w, num_features]
            * masks: 真实的掩码图 [batch_size][num_objs,im_height,im_width]
            * coeff_data: 预测的原型掩码回归系数 [B, H * W * num_prior_boxes, num_mask_coeffs]            
            * gt_box_t： 当前图片真实框的位置 [B, H * W * num_prior_boxes, 4] 
        """
        # 获取原型掩码图的尺寸
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)


        process_gt_bboxes =  cfg.mask_proto_crop

        if cfg.mask_proto_remove_empty_masks:
            # Make sure to store a copy of this because we edit it to get rid of all-zero masks
            pos = pos.clone()

        loss_m = 0
        
        for idx in range(coeff_data.size(0)):
            with torch.no_grad():
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

                if cfg.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()

                if cfg.mask_proto_remove_empty_masks:
                    # Get rid of gt masks that are so small they get downsampled away
                    very_small_masks = (downsampled_masks.sum(dim=(0,1)) <= 0.0001)
                    for i in range(very_small_masks.size(0)):
                        if very_small_masks[i]:
                            pos[idx, idx_t[idx] == i] = 0

                if cfg.mask_proto_reweight_mask_loss:
                    # Ensure that the gt is binary
                    if not cfg.mask_proto_binarize_downsampled_gt:
                        bin_gt = downsampled_masks.gt(0.5).float()
                    else:
                        bin_gt = downsampled_masks

                    gt_foreground_norm = bin_gt     / (torch.sum(bin_gt,   dim=(0,1), keepdim=True) + 0.0001)
                    gt_background_norm = (1-bin_gt) / (torch.sum(1-bin_gt, dim=(0,1), keepdim=True) + 0.0001)

                    mask_reweighting   = gt_foreground_norm * cfg.mask_proto_reweight_coeff + gt_background_norm
                    mask_reweighting  *= mask_h * mask_w

            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            
            if process_gt_bboxes:
                # Note: this is in point-form
                pos_gt_box_t = gt_box_t[idx, cur_pos]

            if pos_idx_t.size(0) == 0:
                continue

            proto_masks = proto_data[idx]
            proto_coef  = coeff_data[idx, cur_pos, :]
            
            # If we have over the allowed number of masks, select a random sample

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]          

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ proto_coef.t()

            if cfg.mask_proto_crop:
                pred_masks = utils.crop(pred_masks, pos_gt_box_t)
            
            if cfg.mask_proto_mask_activation == "sigmoid":
                pre_loss = F.binary_cross_entropy(pred_masks, mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='none')

        
            loss_m += torch.sum(pre_loss)
        
        losses =  loss_m * cfg.mask_alpha / mask_h / mask_w
    
        return losses