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
--config='./train2.conf' \
--save_dir='./model3' \
--use_gpu=1 \
--gpu_id=0 \
--num_passes=200 \
--show_parameter_stats_period=1000 \
--trainer_count=4 \
--log_period=100 \
--dot_period=10 \
--log_error_clipping=false \
--init_model_path='./model2/pass-00073' \
2>&1 | tee './train3.log'

# --init_model_path='./model/pass-00067' \
 
