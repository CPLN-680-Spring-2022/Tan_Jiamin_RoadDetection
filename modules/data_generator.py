"""
the following functions are based on
https://www.kaggle.com/code/pavell3333/unet-massachusetts

binarize_mask takes an input RGB image and convert it to binary format
data_generator generate small batches to feed into the model each time
"""

import numpy as np
import cv2
import os

def binarize_mask(mask, threshold):
  mask_in = np.array(mask)
  mask_out = np.zeros((mask_in.shape[0], mask_in.shape[1], 1))
  np.place(mask_out[:,:,0], mask_in[:,:,0]>=threshold, 1)
  return mask_out

def data_generator(img_dir, label_dir, batch_size, image_shape):
    list_images = os.listdir(img_dir)
    ids_train_split = range(len(list_images))

    while True:
         for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]

            for id in ids_train_batch:
              img_name = img_dir + list_images[id]
              mask_name = label_dir + 'AOI_4_Shanghai_geojson_roads_img' + list_images[id][41:].split('.')[0] + '.png'
  
              img = cv2.imread(img_name)
              img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              img = cv2.resize(img, image_shape, interpolation=cv2.INTER_AREA)
  
              mask = cv2.imread(mask_name)
              mask = cv2.resize(mask, image_shape, interpolation=cv2.INTER_AREA)
              mask = binarize_mask(mask, 150)              
              
              x_batch += [img]
              y_batch += [mask]    
    
            x_batch = np.array(x_batch) / 255.0
            y_batch = np.expand_dims(np.array(y_batch), -1)

            yield x_batch, y_batch