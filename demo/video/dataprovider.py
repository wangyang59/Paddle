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


def hook(settings, src_path, is_generating, file_list, **kwargs):
    # job_mode = 1: training mode
    # job_mode = 0: generating mode
    settings.job_mode = not is_generating
    num_frames = 10
    batch_size = 50
    image_size = 64
    num_digits = 2
    step_length = 0.1

    settings.dataHandler = data_handler.BouncingMNISTDataHandler(
        num_frames, batch_size, image_size, num_digits, step_length, src_path)

    settings.src_dim = image_size * image_size

    if settings.job_mode:
        settings.slots = {
            'source_image_seq': dense_vector_sequence(settings.src_dim),
            'target_image_seq': dense_vector_sequence(settings.src_dim),
            'target_image_seq_next': dense_vector_sequence(settings.src_dim)
        }
    else:
        settings.slots = {
            'source_image_seq': dense_vector_sequence(settings.src_dim),
            'target_image_seq': dense_vector_sequence(settings.src_dim)
        }


@provider(init_hook=hook)
def process(settings, file_name):
    first_frame = np.zeros(settings.src_dim).astype("float32")
    if settings.job_mode == 1:
        if "train" in file_name:
            n = 1000
        else:
            n = 100
    else:
        n = 1
    for i in xrange(n):
        batch = settings.dataHandler.GetBatch()[0]
        for j in xrange(batch.shape[0]):
            seq = list(batch[j].reshape(-1, settings.src_dim))
            if settings.job_mode:
                yield {
                    'source_image_seq': seq[0:5],
                    'target_image_seq': [first_frame] + seq[5:-1],
                    'target_image_seq_next': seq[5:]
                }
            else:
                yield {
                    'source_image_seq': seq[0:5],
                    'target_image_seq': seq[5:]
                }
