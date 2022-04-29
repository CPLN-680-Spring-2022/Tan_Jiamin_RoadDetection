"""
concatenate_split takes the predition which are parse (256,256) images and concatenate them to (1280,1280) images
"""

import os
import cv2
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

def concatenate_split(input_dir, test_num, mask, prediction):
  if prediction is None:
    filename_list = [(input_dir + x) for x in os.listdir(input_dir)]
    front = '/content/drive/MyDrive/School/UPenn/Spring_2022/CPLN_680/sh_all_mask_split/test/masks_xbuffer/AOI_4_Shanghai_geojson_roads_'
    
    if mask == True:
      img_id = []
      for filename in filename_list:
        img_id.append(filename.split('_')[10:][2] + '_' + filename.split('_')[10:][3][0:5])
      filename_list = [(front + x + '.png') for x in img_id]  
    
    split_images = []
    print('loading split images')
    for split_image in tqdm(filename_list):
      img = cv2.imread(split_image)
      split_images.append(img)
  
  if prediction is not None:
    print('loading predictions')
    split_images = prediction
  
  concat_images = []
  print('Concatenating...')
  for wholeTestImage_id in tqdm(range(test_num)):
    h_composite_list = [split_images[x:x+5] for x in range(25*wholeTestImage_id, 25*(wholeTestImage_id+1), 5)]
    im_h = []
    for i in range(len(h_composite_list)):
      im_h.append(cv2.hconcat(h_composite_list[i]))
    im_v = cv2.vconcat(im_h)
    concat_images.append(im_v)

  return concat_images