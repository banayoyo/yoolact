"""
Author: JoshuaQYH

Description:
    本文件主要编写一些训练和预测相关的辅助函数代码
"""
from torch.autograd import Variable 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import random
import numpy as np
import os 


class ScatterWrapper:
    """
        搬运自官方 yolact。其作用是把生成的mask list封装起来，返回可以数据并行的格式
    """
    """ Input is any number of lists. This will preserve them through a dataparallel scatter. """
    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, list):
                print('Warning: ScatterWrapper got input of non-list type.')
        self.args = args
        self.batch_size = len(args[0])
    
    def make_mask(self):
        out = torch.Tensor(list(range(self.batch_size))).long()
        if self.args.cuda: return out.cuda()
        else: return out
    
    def get_args(self, mask):
        device = mask.device
        mask = [int(x) for x in mask]
        out_args = [[] for _ in self.args]

        for out, arg in zip(out_args, self.args):
            for idx in mask:
                x = arg[idx]
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                out.append(x)
        
        return out_args

def prepare_data(datum):
    images, (targets, masks, num_crowds) = datum
    
    if torch.cuda.is_available():
        images = Variable(images.cuda(), requires_grad=False)
        targets = [Variable(ann.cuda(), requires_grad=False) for ann in targets]
        masks = [Variable(mask.cuda(), requires_grad=False) for mask in masks]
    else:
        images = Variable(images, requires_grad=False)
        targets = [Variable(ann, requires_grad=False) for ann in targets]
        masks = [Variable(mask, requires_grad=False) for mask in masks]

    return images, targets, masks, num_crowds

def show_curve(data, xlabel, ylabel, save_path):
    x = [i + 1 for i in range(len(data))]
    y = data
    df = pd.DataFrame({'x': x, 'y': y})
    sns.lineplot(x=xlabel, y=ylabel,datat=df)
    plt.savefig(save_path)


def xy_to_cxcy(xy):
    """
    将bbox的边界坐标（xmin, ymin, xmax, ymax)转化为中心坐标(cx, cy, w, h)
    @params:
        xy: a tensor (nbbox, 4).
    @return:
        a tensor of center coordinate (nbbox, 4)    
    """
    return torch.cat([(xy[:, 2:] + xy[:,:2])/2, (xy[:,2:] - xy[:,:2])], 1)

def cxcy_to_xy(cxcy):
    """
    中心坐标(cx, cy, w, h) 转化为bbox的边界坐标（xmin, ymin, xmax, ymax)
    @params:
        xy: a tensor (nbbox, 4).
    @return:
        a tensor of center coordinate (nbbox, 4)    
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2: ] / 2)],1)

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    encode： cxcy为实际groundtruth的中心坐标值，
    priors_cxcy为先验框坐标,二者经过encode得到二者代表的偏移值，与真实框与先验框的偏移值进行损失函数的计算，box回归
    @params:
        cxcy: (cx, cy, w, h)
        priors_cxcy: (p_cx, p_cy, p_w, p_h)
    @return:
        gcxgcy: (g_cx, g_cy, g_w, g_h) 由gt box与prior box计算bbox偏移值
    
    这三者的计算关系如下：
    g_cx  = (cx - p_cx) / p_cx   g_cy = (cy - p_cy) / p_cy
    g_w = log(w/p_w)  g_h = log(h / p_h)
    """
    variance = [0.1, 0.2] # for 'scaling the localization gradient'
    g_cx_cy = (cxcy[:,:2] - priors_cxcy[:, :2]) / priors_cxcy[:, :2] / variance[0]
    g_w_h = torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) / variance[1]
    return torch.cat([g_cx_cy, g_w_h], 1)  # [numprior, 4]

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    decode： gcxgcy为偏移值
    priors_cxcy为先验框坐标,二者经过decode得到二者预测的真实bbox坐标
    @params:
        gcxgcy: (g_cx, g_cy, g_w, g_h)
        priors_cxcy: (p_cx, p_cy, p_w, p_h)
    @return:
        cxcy: (cx, cy, w, h) 由先验框和偏移值预测的真实bbox坐标
    
    这三者的计算关系如下：
    cx = p_cx * g_cx + p_cx   cy = p_cy * g_cy + p_cy
    w = p_w * exp(g_w)   h = p_h * exp(g_h)
    """
    variance = [0.1, 0.2] # for 'scaling the localization gradient' 
    c_x_y = gcxgcy[:, :2] * priors_cxcy[:,2] * variance[0] + priors_cxcy[:,:2]
    w_h = priors_cxcy[:,2:] * torch.exp(gcxgcy[:, 2:]) * variance[1]
    return torch.cat([c_x_y, w_h],1)


# From https://github.com/amdegroot/ssd.pytorch
def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2, iscrowd=False):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    use_batch = True
    if set_1.dim() == 2:
        use_batch = False
        set_1 = set_1[None, ...]
        set_2 = set_2[None, ...]

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    
    out = intersection / areas_set_1 if iscrowd else intersection/union
    return out  # (n1, n2)


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h

def crop(masks, boxes, padding=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    with torch.no_grad():
        h, w, n = masks.size()
        boxes = boxes.clone() # Some in-place stuff goes on here
        x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=True)
        y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=True)

        rows = torch.arange(w, device=masks.device)[None, :, None].expand(h, w, n)
        cols = torch.arange(h, device=masks.device)[:, None, None].expand(h, w, n)
        
        masks_left  = rows >= x1[None, None, :]
        masks_right = rows <  x2[None, None, :]
        masks_up    = cols >= y1[None, None, :]
        masks_down  = cols <  y2[None, None, :]
        
        crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask.float()


def clip_gradient(optimizer, grad_clip):
    for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    搬运自yolact官方data文件夹中的init文件。
    """
    targets = []
    imgs = []
    masks = []
    num_crowds = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1][0]))
        masks.append(torch.FloatTensor(sample[1][1]))
        num_crowds.append(sample[1][2])

    return torch.stack(imgs, 0), (targets, masks, num_crowds)