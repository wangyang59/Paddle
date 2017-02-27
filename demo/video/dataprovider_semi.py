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
import data_handler_su as data_handler
import numpy as np


def hook(settings, src_path, is_generating, file_list, **kwargs):
    num_frames = 20
    batch_size = 50
    image_size = 64
    num_digits = 1
    step_length = 0.1

    settings.is_generating = is_generating
    settings.dataHandler_train = data_handler.BouncingMNISTDataHandler(
        num_frames, batch_size, image_size, num_digits, step_length, src_path,
        'train_full')

    settings.dataHandler_test = data_handler.BouncingMNISTDataHandler(
        num_frames, batch_size, image_size, num_digits, step_length, src_path,
        'test')

    settings.src_dim = image_size * image_size

    settings.slots = {
        'source_image_seq': dense_vector_sequence(settings.src_dim),
        'target_image_seq': dense_vector_sequence(settings.src_dim),
        'label': integer_value_sequence(10),
        'weight': dense_vector_sequence(1),
        'invWeight': dense_vector_sequence(1)
    }


@provider(init_hook=hook, pool_size=100 * 50)
def process(settings, file_name):
    if "train" in file_name:
        n = 1000
    else:
        if not settings.is_generating:
            n = 200
        else:
            n = 1

    for i in xrange(n):
        if 'train' in file_name:
            batch, label, weight = settings.dataHandler_train.GetBatch()
            #batch[0:10, :] = batch[0:10, :] + np.random.normal(size=(10, batch.shape[1]), scale = 0.2)
        else:
            batch, label, weight = settings.dataHandler_test.GetBatch()
        for j in xrange(batch.shape[0]):
            seq = list(batch[j].reshape(-1, settings.src_dim))
            wgt = weight[j] * 500.0
            invWgt = (1.0 - weight[j]) * 500.0
            yield {
                'source_image_seq': seq[0:10],
                'target_image_seq': seq[10:],
                'label': [label[j]] * 10,
                'weight': [[wgt] for jj in range(10)],
                'invWeight': [[invWgt] for jj in range(10)]
            }
