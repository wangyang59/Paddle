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
import data_handler
import numpy as np


def hook(settings, src_path, file_list, **kwargs):
    num_frames = 20
    batch_size = 50
    image_size = 64
    num_digits = 2
    step_length = 0.1

    settings.dataHandler = data_handler.BouncingMNISTDataHandler(
        num_frames, batch_size, image_size, num_digits, step_length, src_path)

    settings.src_dim = image_size * image_size

    settings.slots = {
        'source_image_seq': dense_vector_sequence(settings.src_dim),
        'target_image': dense_vector(settings.src_dim)
    }


@provider(init_hook=hook)
def process(settings, file_name):
    if "train" in file_name:
        n = 1000
    else:
        n = 100

    for i in xrange(n):
        batch = settings.dataHandler.GetBatch()[0]
        for j in xrange(batch.shape[0]):
            seq = list(batch[j].reshape(-1, settings.src_dim))
            yield {'source_image_seq': seq[0:10], 'target_image': seq[10]}
