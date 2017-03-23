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
from PIL import Image
import os


def hook(settings, file_list, **kwargs):
    settings.image_size = 224
    settings.num_frames = 10

    settings.src_dim = settings.image_size * settings.image_size

    settings.slots = {
        'source_image_seq': dense_vector_sequence(settings.src_dim),
        'target_image_seq': dense_vector_sequence(settings.src_dim)
    }


@provider(init_hook=hook, pool_size=100 * 50)
def process(settings, file_dir):
    half_image_size = settings.image_size / 2
    image_files = os.listdir(file_dir)
    if len(image_files) < settings.num_frames:
        return
    images = []
    for cnt, image_file in enumerate(image_files):
        if cnt == 10:
            break
        img = Image.open(os.path.join(file_dir, image_file))
        half_the_width = img.size[0] / 2
        half_the_height = img.size[1] / 2
        img = img.crop((half_the_width - half_image_size,
                        half_the_height - half_image_size,
                        half_the_width + half_image_size,
                        half_the_height + half_image_size))
        images.append(img.getdata())

    yield {
        'source_image_seq': images[0:(settings.num_frames / 2)],
        'target_image_seq': images[(settings.num_frames / 2):],
    }
