import os
from random import shuffle
from PIL import Image
import numpy as np
import timeit

file_dir = "/home/wangyang59/Data/ILSVRC2016_256/Data/VID/train/ILSVRC2015_VID_train_0000"
image_dirs = os.listdir(file_dir)
shuffle(image_dirs)
image_file_paths = []
for image_dir in image_dirs:
    image_files = os.listdir(os.path.join(file_dir, image_dir))
    image_files.sort()
    if len(image_files) < 6:
        continue
    tmp = [
        os.path.join(file_dir, image_dir, image_file)
        for image_file in image_files
    ]
    for i in range(len(tmp) - 6 + 1):
        image_file_paths.append(tmp[i:(i + 6)])
shuffle(image_file_paths)

start = timeit.default_timer()

images = np.zeros((6, 256 * 256 * 3), dtype=np.float32)
half_image_size = 256 / 2
for i in xrange(128):
    for cnt, image_file in enumerate(image_file_paths[i]):
        img = Image.open(image_file)

        half_the_width = img.size[0] / 2
        half_the_height = img.size[1] / 2
        img = img.crop((half_the_width - half_image_size,
                        half_the_height - half_image_size,
                        half_the_width + half_image_size,
                        half_the_height + half_image_size))
        #np.array(list(img.getdata()))#.reshape((-1), order = 'F') / 255.0
        #images.append(np.array(img.getdata(), dtype=np.float32).reshape((-1), order = 'F') / 255.0)
        #print(img.getdata())
        np.ndarray(buffer=img.getdata(), shape=(256 * 256, 3), dtype=np.float32)
        #images[cnt, :] = np.array(img.getdata(), dtype=np.float32).reshape((-1), order = 'F') / 255.0

        img.close()
        #img = open(image_file)
        #img.read(1)
        #img.close()

stop = timeit.default_timer()

print stop - start
