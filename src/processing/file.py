import os 
import cv2 
import glob 

class Processing:
    def __init__(self, path: str) -> None:
        self.path = path
    
    def change_image_name(self) -> None:
        files = os.listdir(self.path)

        for i, file in enumerate(files):
            os.rename(
                os.path.join(self.path, file), 
                os.path.join(self.path, ''.join([str(i), '.jpg']))
        )

        print('finished')
    
    def resize(self) -> None:
        os.mkdir('resized')
        files = glob.glob(self.path + "/*.jpg")

        i = 0
        for img in files:
            _img = cv2.imread(img)
            _img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'resized/{i}', _img)
            i += 1
        
        print('finished')

if __name__ == '__main__':
    path = r'../GAN-sonmi/data/test'
    Processing.change_image_name(path)
