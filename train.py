"""
Author: JoshuaQYH

Description:
    本代码主要用于训练模型
"""
from __future__ import print_function
from __future__ import division
from torchvision import transforms
import torch
import torch.nn as nn
import yolact_model
import multi_loss
import utils
import coco
import config as cfg
import torch.optim as optim 
import os
import sys
import math
from torch.utils.data import DataLoader
import time


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%d h %d m %d s'%(h, m, s)

def train(model, criterion, dataloader, optimizer, epoches):
    model.train()

    start_time = time.time()
    all_batches = len(dataloader)
    all_losses = []

    try:
        for epoch in range(epoches):

            current_loss = 0
            for idx, data in enumerate(dataloader):
                images, targets, masks, num_crowds = utils.prepare_data(data)

                output = model(images)
                optimizer.zero_grad()

                wrapper = utils.ScatterWrapper(targets, masks, num_crowds)

                losses = criterion(output, wrapper, wrapper.make_mask()) # box\ mask\ conf losses
                
                losses = {k: v for k, v in losses.items()}

                loss = sum([losses[k] for k in losses])

                loss.backward()

                if torch.isfinite(loss).item():
                    optimizer.step()
                else:
                    utils.clip_gradient(optimizer, 1)  # [-1, 1]

                current_loss += loss.item()

                if idx % 100 == 0:
                    print("epoch:[{}/{}],batch:[{}/{}],time:{},loss:{}".format(epoch+1, epoches, idx, all_batches, timeSince(start_time), loss))
            all_losses.append(current_loss)
    except KeyboardInterrupt:
        print("Early stop... save the model")
        model.save_weights(cfg.model_save_path)
        utils.show_curve(all_losses, 'epoch', 'loss value', cfg.loss_img_save_path)

    model.save_weights(cfg.model_save_path)
    utils.show_curve(all_losses, 'epoch', 'loss value', cfg.loss_img_save_path)
    print("Train done!")


if __name__ == "__main__":
    utils.seed_everything()

    device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")

    model = yolact_model.YOlACT("resnet101").to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    criterion = multi_loss.MultiLoss(cfg.num_classes, pos_threshold=cfg.positive_iou_threshold, 
                                    neg_threshold=cfg.negative_iou_threshold, 
                                    negpos_ratio=3, l1_alpha=1).to(device)

    # These are in BGR and are for ImageNet
    transform.append(transforms.Resize(size=(550, 550)))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform)

    dataset = coco.COCODetection(image_path=cfg.train_images,
                            info_file=cfg.train_info,
                            transform=transform)

    data_loader = DataLoader(dataset, cfg.batch_size, num_workers=1, shuffle=True, collate_fn=utils.detection_collate,pin_memory=False)

    train(model, criterion, data_loader, optimizer, cfg.epoches)