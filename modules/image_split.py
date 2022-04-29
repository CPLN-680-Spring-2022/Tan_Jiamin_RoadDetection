"""
the following functions are based on
https://towardsdatascience.com/images-and-masks-splitting-into-multiple-pieces-in-python-with-google-colab-2f6b2ddcb322

randimg generates a list of filenames of randomly selected images from the input directory
get_mask generates a list of filenames of the masks based on the input filename list of images
crop is a generator that defines the desired smaller pieces
split parses images to smaller pieces defined by crop
"""

import os
import sys

import numpy as np
import random
import cv2

from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

def randimg(input_dir, n, seed):
  random.seed(seed)
  id_list = random.sample(range(len(os.listdir(input_dir))), n)
  img_list = [f for f in
              list(map(os.listdir(input_dir).__getitem__, id_list))
              if os.path.isfile(os.path.join(input_dir, f))]
  return img_list

def get_mask(img_list):
  mask_list = []
  for img in img_list:
    mask_list.append('AOI_4_Shanghai_geojson_roads_img' + img.split('.')[0][41:] + '.png')
  return mask_list

def crop(input_file, height, width):
  img = cv2.imread(input_file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_height = img.shape[0]
  img_width = img.shape[1]
  for i in range(img_height//height):
      for j in range(img_width//width):
          ht = i*height
          hb = (i+1)*height
          wl = j*width
          wr = (j+1)*width
          yield img[ht:hb, wl:wr]

def split(input_dir, output_dir, height, width, n, start_num, img_list):
  for infile in tqdm(img_list):
      infile_path = os.path.join(input_dir, infile)
      for n, piece in enumerate(crop(infile_path, height, width), start_num):
          img = piece
          img_path = os.path.join(output_dir, 
                                  infile.split('.')[0]+ '_'
                                  + str(n).zfill(5) + '.' + infile.split('.')[1])
          cv2.imwrite(img_path, img)
      sys.stdout.flush()