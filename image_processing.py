## TODO 

import os 
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

## set file path
filepath = r'*/test pic'
filename = 'tested'

## create a folder to store the converted photos
try:
    os.mkdir('train')
except(FileExistsError):
    pass 
except Exception as e:
    raise e


def check_empty_foler(path: str):
    if len(os.listdir(path)) == 0:
        print('folder is empty')

    else:
        pass 

## apply gray scale and adjust image size -> image dim: width x height x 1 
def cvt_images(path, width: int=80, height: int=80):
    counter = 0

    for img in os.listdir(path):
        check_empty_foler(path)

        _img = cv2.imread(os.path.join(path, img))
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        _img = cv2.resize(_img, (width, height))

        cv2.imwrite('train/%04i.jpg' %counter, _img)

        counter += 1

    print(f'{len(os.listdir(path))} images has been converted!')


if __name__ == '__main__':
    cvt_images(filepath)