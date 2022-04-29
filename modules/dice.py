"""
the following functions are based on
https://www.kaggle.com/code/pavell3333/unet-massachusetts

dice_coef generates the dice coefficient
dice_coef_loss equals to 1-dice_coef, it is used as a loss function
"""

from keras import backend as K

def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  dice_coef = (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)
  return dice_coef
 
def dice_coef_loss(y_true, y_pred):
  dice_coef_loss = 1-dice_coef(y_true, y_pred)
  return dice_coef_loss