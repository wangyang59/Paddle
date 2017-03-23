from paddle.trainer_config_helpers import *
from paddle.trainer_config_helpers.layers import get_output_layer
from util import *

error_clipping = 10.0


def seq_to_image_data(data_dir, is_generating):
    """
    Predefined seqToseq train data provider for application
    is_generating: whether this config is used for generating
    """
    define_py_data_sources2(
        "./videoData/train.list",
        "./videoData/test.list",
        module="dataprovider_video",
        obj="process",
        args={"is_generating": is_generating})


def create_lstm_encode_step(hidden_size, nameIdx):
    def lstm_step(current_input):
        hidden_mem = memory(name='lstm_h' + nameIdx, size=hidden_size)
        cell_mem = memory(name='lstm_c' + nameIdx, size=hidden_size)

        concat = concat_layer(input=[current_input, hidden_mem])

        with mixed_layer(
                size=hidden_size * 4,
                layer_attr=ExtraAttr(
                    error_clipping_threshold=error_clipping)) as ipt:
            ipt += full_matrix_projection(concat)

        lstm_out = lstm_step_layer(
            ipt,
            cell_mem,
            hidden_size,
            act=TanhActivation(),
            name="lstm_h" + nameIdx,
            gate_act=SigmoidActivation(),
            state_act=TanhActivation())

        cell_out = get_output_layer(
            name="lstm_c" + nameIdx, input=lstm_out, arg_name='state')
        return lstm_out, cell_out

    return lstm_step


def create_lstm_decode_step(hidden_size, boot_h, boot_c, nameIdx):
    def lstm_step(input):
        hidden_mem = memory(
            name='lstm_h' + nameIdx, size=hidden_size, boot_layer=boot_h)
        cell_mem = memory(
            name='lstm_c' + nameIdx, size=hidden_size, boot_layer=boot_c)

        with mixed_layer(
                size=hidden_size * 4,
                layer_attr=ExtraAttr(
                    error_clipping_threshold=error_clipping)) as ipt:
            ipt += full_matrix_projection(hidden_mem)

        lstm_out = lstm_step_layer(
            ipt,
            cell_mem,
            hidden_size,
            act=TanhActivation(),
            name="lstm_h" + nameIdx,
            gate_act=SigmoidActivation(),
            state_act=TanhActivation())

        get_output_layer(
            name="lstm_c" + nameIdx, input=lstm_out, arg_name='state')
        return lstm_out

    return lstm_step


def conv_lstm_net(is_generating):
    features_num = 32
    label_size = 10

    src_image = data_layer(name='source_image_seq', size=224 * 224)
    trg_image = data_layer(name='target_image_seq', size=224 * 224)

    inputs(src_image, trg_image)

    param_attr = ParamAttr(initial_mean=0.0, initial_std=0.02)
    bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.0)
    param_attr_bn = ParamAttr(initial_mean=1.0, initial_std=0.02)

    if is_generating:
        outputs(concat_layer(input=[src_image, trg_image]))
    else:
        pass


#         with mixed_layer() as entropy:
#             entropy += dotmul_operator(
#                 id, layer_math.log(id + 1e-10), scale=-1.0)
#         cost5 = sum_cost(entropy) * invWgt
#         #         
#         #         id_bn = batch_norm_layer(
#         #             id,
#         #             bias_attr=bias_attr,
#         #             param_attr=param_attr_bn,
#         #             use_global_stats=False)
#         #         
#         #         shape_bn = batch_norm_layer(
#         #             shape,
#         #             bias_attr=bias_attr,
#         #             param_attr=param_attr_bn,
#         #             use_global_stats=False)
#         #         
#         #         pose_bn = batch_norm_layer(
#         #             pose,
#         #             bias_attr=bias_attr,
#         #             param_attr=param_attr_bn,
#         #             use_global_stats=False)
#         #         
#         #         cost4 = sum_cost(out_prod_layer(id_bn, shape_bn)) \
#         #                 + sum_cost(out_prod_layer(id_bn, pose_bn)) \
#         #                 + sum_cost(out_prod_layer(shape_bn, pose_bn))
# 
#         cost3 = classification_cost(input=id, label=lbl, weight=wgt)
#         cost1 = regression_cost(input=future, label=trg_image)
#         cost2 = regression_cost(
#             input=recon,
#             label=expand_layer(
#                 input=first_seq(src_image), expand_as=src_image))
# 
#         sum_evaluator(cost1, name="cost1")
#         sum_evaluator(cost2, name="cost2")
#         sum_evaluator(cost3, name="cost3")
#         #         sum_evaluator(cost4, name="cost4")
#         sum_evaluator(cost5, name="cost5")
#         outputs(cost1, cost2, cost3, cost5)
