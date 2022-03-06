## converting multiple images with cv2
## test folder : test pic
## test save folder : train 

### TODO
# 1. add some parser
# 2. colored text for warning(red) or finished(green)
# 3. convert to numpy array and load orioginal images

import os
from typing import List 
import cv2 
import argparse
import numpy as np
import matplotlib.pyplot as plt 

"""parser desc. 

parser = argparse.ArgumentParser(description='Converting image(s)')
parser.add_argument(
                '-i', '--image',
                required=True,
                help='add target folder'
            )
args = vars(parser.parse_args())

parser_path = args['image']

"""


## set file path
filepath = r'/Users/not_joon/projects/GAN-sonmi/junk/test pic'
savepath = r'/Users/not_joon/projects/GAN-sonmi/img_np_array'

## input files are must have these foramt
ALLOW_EXTS = {'.jpg', '.jpeg', '.png'}


def make_cvt_dir(folder_name: str='train'):
    ## create a folder to store converted images
    try:
        os.mkdir(folder_name)

    except(FileExistsError):
        pass 

    except Exception as e:
        raise e


def check_empty_foler(path: str):
    if len(os.listdir(path)) == 0:
        print('!!! folder is empty !!!')

    else:
        pass 


def check_valid_extention(filename):

    # get file extenstion '~.jpg'
    exts = os.path.splitext(filename)[1]

    if exts not in ALLOW_EXTS:
        print(f'!!! Invalide file extension >> {exts} <<< has decteted !!!')

    else:
        pass


def cvt_and_resize_imgs(path, width: int=64, height: int=64) -> None:
    ## apply gray scale and adjust image size 
    ## expected image dim: width x height x 1 
    counter = 0
    
    ## https://towardsdatascience.com/loading-custom-image-dataset-for-deep-learning-models-part-1-d64fa7aaeca6
    _img_to_array = []

    for img in os.listdir(path):
        filename = os.path.join(path, img)

        check_valid_extention(filename)

        _img = cv2.imread(filename)
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        _img = cv2.resize(_img, (width, height), interpolation=cv2.INTER_AREA)

        _img_to_array.append([_img])
        np.save(os.path.join(savepath, 'features'), np.array(_img_to_array))

        """
        cv2.imwrite(f'train/{counter}.jpeg', _img)
        counter += 1
        """

    print(f'DONE: {len(os.listdir(path))} images has been converted!')

def load_npy_file(filepath: str = savepath):
    if len(os.listdir(filepath)) <= 0:
        print("!!! NO FILE(S) IN THE DIRECTORY")
        
    else:
        _load_file = np.load(os.path.join(filepath, 'features.npy'))
    
    ## !! TEST 
    plt.imshow(_load_file[0].reshape(64, 64, 1))


def __len__(filepath:str = savepath) -> int:
    return len(filepath)

    
if __name__ == '__main__':
    make_cvt_dir('train')
    check_empty_foler(filepath)
    cvt_and_resize_imgs(filepath)
    load_npy_file(savepath)