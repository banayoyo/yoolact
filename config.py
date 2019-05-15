"""
Author：JoshuaQYH

Description：
    本文件主要用于定义一些参数常量，将多个文件中的重要参数常量统一在同一个文件中，方便更改
"""

DEVICE_ID = 3
epoches = 30
batch_size = 16
lr =2 * 1e-3
momentum = 0.9

model_save_path = "yolact-v1.pth"
loss_img_save_path = 'loss-v1.png'

train_images = "./data/coco/images"
train_info = 'path_to_annotation_file'

# COCO 数据集有80类 + 1类 （1为背景）
num_classes = 81 

# ----------- 损失函数的中阈值 -------------- #
positive_iou_threshold = 0.5
negative_iou_threshold = 0.5
crowd_iou_threshold = 1

bbox_alpha = 1

focal_loss_alpha = 0.2
focal_loss_gamma = 2
conf_alpha = 1


mask_proto_remove_empty_masks = False
mask_proto_binarize_downsampled_gt = True
mask_proto_reweight_mask_loss = False
mask_proto_double_loss = False
mask_proto_double_loss_alpha = 1
mask_proto_crop = True
masks_to_train = 100
mask_proto_mask_activation = "sigmoid"
mask_proto_normalize_mask_loss_by_sqrt_area = False
mask_proto_normalize_emulate_roi_pooling = True

#------------- backbone 配置 ------------- #

resnet101 = {
    'layers': [3,4,23,3],   # 用于定义resnet的层数
    'selected_layers': list(range(0,5)) # 用于定义resnet中用于特征金字塔的特征层
}

resnet50 = {
    'layers': [3,4,6,3],
    'selected_layer': list(range(0,5))
}

pred_setting = {
    'pred_scales': [[1]],  # 预测的先验框的尺寸
    'pred_aspect_ratios': [0.66, 0.5, 1, 1.5, 2] # 预测先验框的长宽比，来自yolact
}

fpn = {
    'num_feature': 256,  # 每一卷积层输出通道数
    'num_downsample': 2  # fpn中降采样的层数  
}

num_mask_coeff = 64
num_mask_proto = 64

# ---------------- copy from official yolact config.py

# for making bounding boxes pretty
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}




