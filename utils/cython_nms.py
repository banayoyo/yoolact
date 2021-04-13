## Note: Figure out the license details later.
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

#import cython
import numpy as np

def max(a, b):
    return a if a >= b else b

def min(a, b):
    return a if a <= b else b

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    #ii, jj
    # sorted indices
    #i, j
    # temp variables for box i's (the box currently under consideration)
    #ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    #xx1, yy1, xx2, yy2
    #w, h
    #inter, ovr

    for ii in range(ndets):
      i = order[ii]
      if suppressed[i] == 1:
          continue
      ix1 = x1[i]
      iy1 = y1[i]
      ix2 = x2[i]
      iy2 = y2[i]
      iarea = areas[i]
      for jj in range(ii + 1, ndets):
          j = order[jj]
          if suppressed[j] == 1:
              continue
          xx1 = max(ix1, x1[j])
          yy1 = max(iy1, y1[j])
          xx2 = min(ix2, x2[j])
          yy2 = min(iy2, y2[j])
          w = max(0.0, xx2 - xx1 + 1)
          h = max(0.0, yy2 - yy1 + 1)
          inter = w * h
          ovr = inter / (iarea + areas[j] - inter)
          if ovr >= thresh:
              suppressed[j] = 1

    return np.where(suppressed == 0)[0]
