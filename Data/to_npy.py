import os
import time

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

target = r'...'

if not os.path.isfile(target):
    print("loading data...")
    start = time.time()

    data = []
    path = os.join.path(target)

    for file in tqdm(os.listdir(path)):
        f = os.path.join(path, file)
        image = Image.open(f).resize((64, 64), Image.ANTIALIAS).convert('RGB')
        data.append(np.asarray(image))
    
    data = np.reshape(data, (len(data), 64, 64, 3)) #(N, H, W, C)
    data = data.astype(np.float32)
    data = data / 255.0

    print("saving data...")
    np.save(target, data)
    end = time.time()
    print("done. elapsed {} seconds".format(end - start))

else:
    print("loading data...")
    data = np.load(target)
    print("done.")