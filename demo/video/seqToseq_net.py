# edit-mode: -*- python -*-

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

import sys
import os
from paddle.trainer_config_helpers import *


def seq_to_seq_data(data_dir, is_generating):
    """
    Predefined seqToseq train data provider for application
    is_generating: whether this config is used for generating
    """
    define_py_data_sources2(
        "./data/train.list",
        "./data/test.list",
        module="dataprovider",
        obj="process",
        args={"src_path": data_dir,
              "is_generating": is_generating})


def gru_encoder_decoder(is_generating,
                        img_embed_dim=1024,
                        encoder_size=1024,
                        decoder_size=1024,
                        beam_size=3,
                        max_length=250):
    """
    A wrapper for an attention version of GRU Encoder-Decoder network
    is_generating: whether this config is used for generating
    encoder_size: dimension of hidden unit in GRU Encoder network
    decoder_size: dimension of hidden unit in GRU Decoder network
    word_vector_dim: dimension of word vector
    beam_size: expand width in beam search
    max_length: a stop condition of sequence generation
    """
    # Declare inputs at the beginning
    src_image = data_layer(name='source_image_seq', size=64 * 64)
    trg_image = data_layer(name='target_image_seq', size=64 * 64)
    if not is_generating:
        trg_next_image = data_layer(name='target_image_seq_next', size=64 * 64)
        inputs(src_image, trg_image, trg_next_image)
    else:
        inputs(src_image, trg_image)

    src_embedding = fc_layer(
        input=src_image,
        name="src_embedding",
        size=img_embed_dim,
        bias_attr=ParamAttr(name="_src_embedding_bias"),
        param_attr=ParamAttr(name="_src_embedding_param"),
        act=ReluActivation())

    src_forward = simple_gru(input=src_embedding, size=encoder_size)
    src_backward = simple_gru(
        input=src_embedding, size=encoder_size, reverse=True)
    encoded_vector = concat_layer(input=[src_forward, src_backward])

    with mixed_layer(size=decoder_size) as encoded_proj:
        encoded_proj += full_matrix_projection(input=encoded_vector)

    backward_first = first_seq(input=src_backward)
    with mixed_layer(size=decoder_size, act=TanhActivation()) as decoder_boot:
        decoder_boot += full_matrix_projection(input=backward_first)

    def gru_decoder_with_attention(enc_vec, enc_proj, current_img_embed):
        decoder_mem = memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)

        context = simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        with mixed_layer(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += full_matrix_projection(input=context)
            decoder_inputs += full_matrix_projection(input=current_img_embed)

        gru_step = gru_step_layer(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        with mixed_layer(
                size=64 * 64, bias_attr=True, act=ReluActivation()) as out:
            out += full_matrix_projection(input=gru_step)
        return out

    def gru_decoder_with_attention_gen(enc_vec, enc_proj, current_img_embed):
        decoder_mem = memory(
            name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)
        img_gen_mem = memory(name='gru_img_gen', size=64 * 64)

        img_gen_embed_mem = fc_layer(
            input=img_gen_mem,
            name="gru_img_embed_gen",
            size=img_embed_dim,
            bias_attr=ParamAttr(name="_src_embedding_bias"),
            param_attr=ParamAttr(name="_src_embedding_param"),
            act=ReluActivation())

        context = simple_attention(
            encoded_sequence=enc_vec,
            encoded_proj=enc_proj,
            decoder_state=decoder_mem)

        with mixed_layer(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += full_matrix_projection(input=context)
            decoder_inputs += full_matrix_projection(input=img_gen_embed_mem)

        gru_step = gru_step_layer(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        out = mixed_layer(
            size=64 * 64,
            bias_attr=True,
            act=ReluActivation(),
            input=full_matrix_projection(input=gru_step))

        out2 = mixed_layer(
            size=64 * 64,
            act=IdentityActivation(),
            input=identity_projection(input=out),
            name="gru_img_gen")
        # decoder_mem.set_input(out)
        return out

    decoder_group_name = "decoder_group"
    group_inputs = [
        StaticInput(
            input=encoded_vector, is_seq=True), StaticInput(
                input=encoded_proj, is_seq=True)
    ]

    if not is_generating:
        trg_embedding = fc_layer(
            input=trg_image,
            name="trg_embedding",
            size=img_embed_dim,
            bias_attr=ParamAttr(name="_src_embedding_bias"),
            param_attr=ParamAttr(name="_src_embedding_param"),
            act=ReluActivation())
        group_inputs.append(trg_embedding)

        # For decoder equipped with attention mechanism, in training,
        # target embeding (the groudtruth) is the data input,
        # while encoded source sequence is accessed to as an unbounded memory.
        # Here, the StaticInput defines a read-only memory
        # for the recurrent_group.
        decoder = recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_with_attention,
            input=group_inputs)

        cost = regression_cost(input=decoder, label=trg_next_image)
        outputs(cost)
    else:
        # In generation, the decoder predicts a next target word based on
        # the encoded source sequence and the last generated target word.

        # The encoded source sequence (encoder's output) must be specified by
        # StaticInput, which is a read-only memory.
        # Embedding of the last generated word is automatically gotten by
        # GeneratedInputs, which is initialized by a start mark, such as <s>,
        # and must be included in generation.

        trg_embedding = fc_layer(
            input=trg_image,
            name="trg_embedding",
            size=img_embed_dim,
            bias_attr=ParamAttr(name="_src_embedding_bias"),
            param_attr=ParamAttr(name="_src_embedding_param"),
            act=ReluActivation())
        group_inputs.append(trg_embedding)
        # For decoder equipped with attention mechanism, in training,
        # target embeding (the groudtruth) is the data input,
        # while encoded source sequence is accessed to as an unbounded memory.
        # Here, the StaticInput defines a read-only memory
        # for the recurrent_group.
        decoder = recurrent_group(
            name=decoder_group_name,
            step=gru_decoder_with_attention_gen,
            input=group_inputs)

        outputs(decoder)
