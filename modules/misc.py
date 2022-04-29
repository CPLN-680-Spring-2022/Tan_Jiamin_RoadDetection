"""
the following functions are based on
https://www.kaggle.com/code/pavell3333/unet-massachusetts

convert_to_bw takes an input image in binary arrays and convert it to RGB image with black and white colors
plot_test_pred draws the following in the order of: test image, test mask, prediction

"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

def convert_to_bw(input_image_list, test_num, threshold = 0.5):
  height= input_image_list[0].shape[0]
  width= input_image_list[0].shape[1]
  output = np.zeros((test_num, height, width, 3))
  for i in tqdm(range(test_num)):
    for j in range(height):
      for k in range(width):
        if input_image_list[i][j][k] > threshold:
          output[i,j,k,[0, 1, 2]] = 255
  return output

def plot_test_pred(image, mask, prediction, test_num):
  for i in range(test_num):
    plt.figure(figsize = (20, 600));
    plt.subplot(test_num, 3, i*3+1).imshow(image[i]);
    plt.title('Test Image');
    plt.axis('off');
    
    plt.subplot(test_num, 3, i*3+2).imshow(np.squeeze(mask[i]));
    plt.title('Test Mask');
    plt.axis('off');
    
    plt.subplot(test_num, 3, i*3+3).imshow(prediction[i]/255);
    plt.title('Prediction');
    plt.axis('off');
    
    plt.show();