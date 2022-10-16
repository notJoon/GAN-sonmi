import os 
import fnmatch
import cv2 
import numpy as np

PATH = r'/Users/not_joon/Desktop/naver1/'

class ImageProcessingTools:
    def __init__(self, path: str) -> None:
        self.path = path
        self.path_len = len(os.listdir(path))
        self.format = "*" + ".png"
        self.files = fnmatch.filter(os.listdir(self.path), self.format)
        self.image_size = (128, 128)
        self.channels = 1 if self.grayscale else 3
        # self.classes = 44

    def rename(self, start_num: int) -> str:
        for i, file in enumerate(self.files):
            num = i + start_num
            _file = str(num) + self.format

            os.rename(self.path + file, self.path + _file)

        print(f"{self.path_len} of files are converted.")

        return self 
    
    def resize(self) -> str:
        for _, file in enumerate(self.files):
            img = cv2.imread(self.path + file)
            img = cv2.resize(img, self.image_size)
            cv2.imwrite(self.path + file, img)

        print(f"{self.path_len} of files are converted.")

        return self

    def greyscale(self) -> str:
        for _, file in enumerate(self.files):
            img = cv2.imread(self.path + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(self.path + file, img)

        print(f"{self.path_len} of files are converted.") 
        
        return self 

    # ref: https://github.com/foolmarks/images_to_npy/blob/master/img_to_npy.py#L33
    def images_to_npy(self, normalize=False, labels=False, compress=True) -> None:
        f = open(self.path, 'r')
        images = f.readlines()
        f.close()

        if normalize:
            x = np.ndarray(
                    shape=(len(images), 
                    self.image_size[0], 
                    self.image_size[1], 
                    self.channels), 
                    dtype=np.float32,
                    order='C',
                )
        else:
            x = np.ndarray(
                    shape=(len(images), 
                    self.image_size[0], 
                    self.image_size[1], 
                    self.channels), 
                    dtype=np.uint8,
                    order='C',
                )
        
        for i in range(len(images)):
            img = cv2.imread(os.join(self.path, i + self.format))

            if normalize:
                x[i] = (img / 255.0).astype(np.float32)
            else:
                x[i] = img



if __name__ == "__main__":
    pass 