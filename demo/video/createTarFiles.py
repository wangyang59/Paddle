import tarfile
import os
from random import shuffle

file_dir = "/home/wangyang59/Data/ILSVRC2016_256/Data/VID/val"
train_dirs = os.listdir(file_dir)

all_image_dirs = []
# for train_dir in train_dirs:
#     image_dirs = os.listdir(os.path.join(file_dir, train_dir))
#     for image_dir in image_dirs:
#         all_image_dirs.append(os.path.join(file_dir, train_dir, image_dir))
for train_dir in train_dirs:
    all_image_dirs.append(os.path.join(file_dir, train_dir))

shuffle(all_image_dirs)

num_tars = len(all_image_dirs) / 100

for i in range(num_tars):
    tar = tarfile.open(
        "/home/wangyang59/Data/ILSVRC2016_256_tar/val/val_%s.tar" % i, "w")
    for j in range(100):
        if len(all_image_dirs) == 0:
            break
        tmp = all_image_dirs.pop()
        tar.add(tmp)
    tar.close()
    print("created /home/wangyang59/Data/ILSVRC2016_256_tar/val/val_%s.tar" % i)
