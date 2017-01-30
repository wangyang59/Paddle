#!/bin/bash
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
set -e

paddle train \
    --job=test \
    --config='gen2.conf' \
    --use_gpu=1 \
    --trainer_count=1 \
    --init_model_path='./model2/pass-00071' \
    --predict_output_dir=. \
    2>&1 | tee 'gen.log'

python visualize.py 20
