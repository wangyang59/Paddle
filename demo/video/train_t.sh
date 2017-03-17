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
--config='./train_t.conf' \
--save_dir='./model_t' \
--use_gpu=1 \
--gpu_id=0 \
--num_passes=500 \
--show_parameter_stats_period=1000 \
--trainer_count=1 \
--log_period=100 \
--dot_period=10 \
--log_error_clipping=false \
--init_model_path='./model_ent5/pass-00131' \
--load_missing_parameter_strategy='rand' \
2>&1 | tee './train_t.log'
