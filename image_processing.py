## TODO 

import os 
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

## set file path
path = '...'

## apply gray scale and adjust image size -> image dim: 80x80x1
imgs = []

for img in os.listdir(path):
    _img = cv2.imread(os.path.join(path, img))
    _img = cv2.cvtColor(_img, cv2.COLOR_BG2BGR)
    _img = cv2.resize(img, (80, 80))
    imgs.append([_img])

## converting the list to numpy array 
np.save(os.path.join(path, 'features'), np.array(imgs))


