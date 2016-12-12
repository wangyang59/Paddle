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

def hook(settings, src_path, is_generating, file_list, **kwargs):
    # job_mode = 1: training mode
    # job_mode = 0: generating mode
    settings.job_mode = not is_generating
    image_size = 64
    
    settings.src_dim =  image_size * image_size
    
    if settings.job_mode:
        settings.slots = {
            'source_image_seq':
            dense_vector_sequence(settings.src_dim),
            'target_image_seq':
            dense_vector_sequence(settings.src_dim),
            'target_image_seq_next':
            dense_vector_sequence(settings.src_dim)
        }
    else:
        settings.slots = {
            'source_language_word':
            integer_value_sequence(len(settings.src_dict)),
            'sent_id':
            integer_value_sequence(len(open(file_list[0], "r").readlines()))
        }
    
    settings.logger.info("src dim %d" % (settings.src_dim))

@provider(init_hook=hook)
def process(settings, file_name):
    settings.logger.info("started process")
    num_frames = 10
    batch_size = 50
    image_size = 64
    num_digits = 2
    step_length = 0.1
    
    dataHandler = data_handler.BouncingMNISTDataHandler(num_frames, batch_size, image_size, 
                                            num_digits, step_length, "./data/mnist.h5")
    

    for i in xrange(100):
        batch = dataHandler.GetBatch()[0]
        for j in xrange(batch.shape[0]):
            seq = map(list, list(batch[j].reshape(-1, settings.src_dim)))
            settings.logger.info("generated %d %d" % (i, j))
            yield {
                'source_image_seq': seq,
                'target_image_seq': seq[0:-1],
                'target_image_seq_next': seq[1:]
            }
            settings.logger.info("generated %d %d" % (i, j))
