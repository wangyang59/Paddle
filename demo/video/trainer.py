# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import argparse
import numpy

from paddle.trainer.config_parser import parse_config
from paddle.trainer.config_parser import logger
import py_paddle.swig_paddle as api


def main():
    api.initPaddle('--use_gpu=1', '--dot_period=10', '--log_period=100',
                   '--save_dir=' + "./model")

    conf = parse_config("train.conf", "")
    logger.info(str(conf.model_config))


if __name__ == '__main__':
    main()
