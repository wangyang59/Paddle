import h5py
import os
from PIL import Image
import numpy as np
import multiprocessing
import functools

file_dir = "/home/wangyang59/Data/ILSVRC2016_256/Data/VID/train/ILSVRC2015_VID_train_0001"
image_dirs = os.listdir(file_dir)

#shuffle(image_dirs)


def dump_h5py(file_dir, image_dir):
    half_image_size = 256 / 2

    image_files = os.listdir(os.path.join(file_dir, image_dir))
    image_files.sort()

    images = np.zeros((len(image_files), 256 * 256 * 3), dtype=np.float32)
    for cnt, image_file in enumerate(image_files):
        img = Image.open(os.path.join(file_dir, image_dir, image_file))
        half_the_width = img.size[0] / 2
        half_the_height = img.size[1] / 2
        img = img.crop((half_the_width - half_image_size,
                        half_the_height - half_image_size,
                        half_the_width + half_image_size,
                        half_the_height + half_image_size))
        #np.array(list(img.getdata()))#.reshape((-1), order = 'F') / 255.0
        #images.append(np.array(img.getdata(), dtype=np.float32).reshape((-1), order = 'F') / 255.0)
        images[cnt, :] = np.array(
            img.getdata(), dtype=np.float32).reshape(
                (-1), order='F') / 255.0

        img.close()

    h5f = h5py.File(
        "/home/wangyang59/Data/ILSVRC2016_h5/train/" + image_dir + '.h5', 'w')
    h5f.create_dataset('data', data=images)
    h5f.close()
    print(image_dir)


fun = functools.partial(dump_h5py, file_dir)
pool = multiprocessing.Pool(10)
pool.imap_unordered(fun, image_dirs, chunksize=10)
pool.close()
pool.join()
#dump_h5py(file_dir, image_dirs[0])
