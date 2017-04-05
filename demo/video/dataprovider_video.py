# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.trainer.PyDataProvider2 import *
import os
from random import shuffle
from image_mltiproc import MultiProcessImageTransformer
import itertools
import h5py
from Queue import Queue
from threading import Thread
import numpy as np


def hook(settings, file_list, **kwargs):
    settings.image_size = 256
    settings.image_channel = 3
    settings.num_frames = 6
    settings.jump = 1
    settings.procnum = 2

    settings.src_dim = settings.image_size * settings.image_size * settings.image_channel
    settings.transformer = MultiProcessImageTransformer(
        procnum=settings.procnum,
        image_size=settings.image_size,
        min_frame=settings.num_frames)
    settings.slots = {
        'source_image_seq': dense_vector_sequence(settings.src_dim),
        'target_image_seq': dense_vector_sequence(settings.src_dim)
    }


#@provider(init_hook=hook, pool_size=1024)
def process(settings, file_dir):
    #     image_dirs = os.listdir(file_dir)
    #     shuffle(image_dirs)
    #     image_file_paths = []
    #     for image_dir in image_dirs:
    #         image_files = os.listdir(os.path.join(file_dir, image_dir))
    #         image_files.sort()
    #         if len(image_files) < 6:
    #             continue
    #         tmp = [os.path.join(file_dir, image_dir, image_file) for image_file in image_files]
    #         for i in range(len(tmp) - 6 + 1):
    #             image_file_paths.append(tmp[i:(i+6)])        
    #     shuffle(image_file_paths)

    image_files = os.listdir(file_dir)
    shuffle(image_files)
    image_file_paths = [
        os.path.join(file_dir, image_file) for image_file in image_files
    ]

    for image_file_path in image_file_paths:
        f = h5py.File(image_file_path, 'r')
        images = f["data"].value
        n = images.shape[0]
        for i in range(n - 6 + 1):
            tmp = list(images[i:(i + 6), :])
            yield {'source_image_seq': tmp[0:5], 'target_image_seq': tmp[5:6]}
            del tmp
        del images
        f.close()


# class EndSignal():
#     pass
#  
# end = EndSignal()
#  
# def read_worker(r, q):
#     for d in r:
#         q.put(d)
#     q.put(end)
#  
# @provider(init_hook=hook, pool_size=1024)       
# def process(settings, file_dir):
#     r = reader(settings, file_dir)
#     q = Queue(maxsize=1024)
#     t = Thread(
#         target=read_worker, args=(
#             r,
#             q, ))
#     t.daemon = True
#     t.start()
#     
#     r = reader(settings, file_dir)
#     t = Thread(
#         target=read_worker, args=(
#             r,
#             q, ))
#     t.daemon = True
#     t.start()
#     
#     e = q.get()
#     while e != end:
#         yield e
#         e = q.get()

#     settings.transformer.run(image_file_paths)
#     
#     end_cnts = 0
#     
#     while end_cnts < settings.procnum:
#         images = list(settings.transformer.q.get())
#         if type(images[0]) is str:
#             end_cnts += 1
#         else:
#             yield{
#                  'source_image_seq': images[0:5],
#                  'target_image_seq': images[5:6]
#                   }
#         del images

#     for i in xrange(len(image_file_paths)):
#         images = list(settings.transformer.q.get())
#         n = len(images)
#         for j in range(n - 6 + 1):
#             yield{
#                 'source_image_seq': images[j:(j+5)],
#                 'target_image_seq': images[(j+5):(j+6)]
#                 }
#         
#         del images

#print(settings.transformer.q.get())
#images = results.next()
#         images = list(settings.transformer.q.get())
#         yield {
#                  'source_image_seq': images[0:5],
#                  'target_image_seq': images[5:6]
#               }
#         del images
#for image_file_paths_chunk in iter(lambda: list(itertools.islice(image_file_paths, 1000)), []):
# for images in settings.transformer.run(image_file_paths):
#     images = list(images)
#     print(settings.transformer.q.qsize())
#     settings.transformer.q.get()
#     yield {
#                  'source_image_seq': images[0:5],
#                  'target_image_seq': images[5:6]
#           }
