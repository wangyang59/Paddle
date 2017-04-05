import os
import h5py
from random import shuffle
import timeit
import numpy as np

file_dir = "/home/wangyang59/Data/ILSVRC2016_h5/train/"
image_files = os.listdir(file_dir)
image_file_paths = [
    os.path.join(file_dir, image_file) for image_file in image_files
]
shuffle(image_file_paths)

cnt = 0
start = timeit.default_timer()

# for file_path in image_file_paths:
# #         f = h5py.File(file_path, 'r')
# #         q.put(f["data"].value)
# #         f.close()
#     f = h5py.File(file_path, 'r')
#     images = f["data"].value
#     n = images.shape[0]
#     for i in range(n - 6 + 1):
#         tmp = list(images[i:(i+6), :])
#         cnt += 1
#         if cnt == 1280:
#             break
#     del images
#     
#     if cnt == 1280:
#         break

for i in range(1280):
    np.random.rand(6, 256 * 256 * 3)

stop = timeit.default_timer()
print stop - start
