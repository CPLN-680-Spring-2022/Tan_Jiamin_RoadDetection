"""
the following functions are based on
https://www.kaggle.com/code/pavell3333/unet-massachusetts

binarize_mask takes an input RGB image and convert it to binary format
F1_score derives the F1_score after predictions are made.
"""

import numpy as np

def binarize_mask(mask, threshold):
  mask_in = np.array(mask)
  mask_out = np.zeros((mask_in.shape[0], mask_in.shape[1], 1))
  np.place(mask_out[:,:,0], mask_in[:,:,0]>=threshold, 1)
  return mask_out

def F1_score(mask, prediction):
  TP = 0
  FN = 0
  FP = 0
  F1_score = 0

  for i in zip(mask, prediction):
    mask = binarize_mask(i[0], 150)
    pred = binarize_mask(i[1], 255)

    try:
      TP_ = np.unique((pred*mask), return_counts=True)[1][1]
    except IndexError:
      TP_ = 0
    try:
      FN_ = np.unique((mask != pred) & (mask == 1), return_counts=True)[1][1]
    except IndexError:
      FN_ = 0
    try:
      FP_ = np.unique((mask != pred) & (mask == 0), return_counts=True)[1][1]
    except IndexError:
      FP_ = 0

    TP = TP + TP_
    FN = FN + FN_
    FP = FP + FP_

  try:
    F1_score = TP/(TP + 0.5*(FP+FN))
  except ZeroDivisionError:
    F1_score = 0
  
  return F1_score