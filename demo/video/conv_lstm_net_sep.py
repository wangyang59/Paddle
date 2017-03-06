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
        "./data/train.list",
        "./data/test.list",
        module="dataprovider_semi",
        obj="process",
        args={"src_path": data_dir,
              "is_generating": is_generating})


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

    src_image = data_layer(name='source_image_seq', size=64 * 64)
    trg_image = data_layer(name='target_image_seq', size=64 * 64)
    lbl = data_layer(name="label", size=label_size)
    wgt = data_layer(name='weight', size=1)
    invWgt = data_layer(name='invWeight', size=1)
    inputs(src_image, trg_image, lbl, wgt, invWgt)

    param_attr = ParamAttr(initial_mean=0.0, initial_std=0.02)
    bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.0)
    param_attr_bn = ParamAttr(initial_mean=1.0, initial_std=0.02)

    conv1 = conv_bn(
        src_image,
        channels=1,
        imgSize=64,
        num_filters=features_num,
        output_x=32,
        stride=2,
        name="conv1",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    conv2 = conv_bn(
        conv1,
        channels=features_num,
        imgSize=32,
        num_filters=features_num * 2,
        output_x=16,
        stride=2,
        name="conv2",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    conv3 = conv_bn(
        conv2,
        channels=features_num * 2,
        imgSize=16,
        num_filters=features_num * 4,
        output_x=8,
        stride=2,
        name="conv3",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    conv4 = conv_bn(
        conv3,
        channels=features_num * 4,
        imgSize=8,
        num_filters=features_num * 8,
        output_x=4,
        stride=2,
        name="conv4",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    hidden1 = fc_layer(
        input=conv4,
        size=features_num * 2,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    pose = fc_layer(
        input=hidden1,
        #size=features_num * 2,
        size=2,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    #    pooled3 = img_pool_layer(
    #        input=conv3,
    #        num_channels=features_num * 4,
    #        stride=1,
    #        pool_size=8,
    #        pool_type=MaxPooling())

    #    hidden4 = fc_layer(
    #        input=pooled3,
    #        size=features_num,
    #        bias_attr=bias_attr,
    #        param_attr=param_attr,
    #        act=ReluActivation())    

    id = fc_layer(
        input=hidden1,
        size=10,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=SoftmaxActivation())

    shape = fc_layer(
        input=hidden1,
        size=10,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    lstm_encode_step1 = create_lstm_encode_step(features_num * 2, "1")
    encoder_layer1, cell_layer1 = recurrent_group(
        name="encoder_1", step=lstm_encode_step1, input=pose)

    lstm_decode_step2 = create_lstm_decode_step(
        features_num * 2,
        boot_h=last_seq(encoder_layer1),
        boot_c=last_seq(cell_layer1),
        nameIdx="2")
    decoder_layer2 = recurrent_group(
        name="decoder_2", step=lstm_decode_step2, input=trg_image)

    new_pose = fc_layer(
        input=decoder_layer2,
        #size=features_num * 2,
        size=2,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    #     hidden2 = tensor_layer(a=id, 
    #                            b=new_pose, 
    #                            size=features_num * 2,
    #                            param_attr=ParamAttr(initial_mean=0.0, initial_std=0.02, name="tensor_w"),
    #                            bias_attr=ParamAttr(initial_mean=0.0, initial_std=0.02, name="tensor_b"))
    #     
    #     hidden3 = tensor_layer(a=id, 
    #                            b=expand_layer(input=first_seq(pose),
    #                                           expand_as=pose), 
    #                            size=features_num * 2,
    #                            param_attr=ParamAttr(initial_mean=0.0, initial_std=0.02, name="tensor_w"),
    #                            bias_attr=ParamAttr(initial_mean=0.0, initial_std=0.02, name="tensor_b"))
    hidden2 = concat_layer([id, shape, new_pose])
    hidden3 = concat_layer(
        [id, shape, expand_layer(
            input=first_seq(pose), expand_as=pose)])

    def deconv(ipt, name):
        #         hidden5 = fc_layer(
        #             input=ipt,
        #             size=features_num * 2,
        #             bias_attr=ParamAttr(
        #                 initial_mean=0.0, initial_std=0.0, name="fc1_b"),
        #             param_attr=ParamAttr(
        #                 initial_mean=0.0, initial_std=0.02, name="fc1_w"),
        #             act=ReluActivation())

        hidden4 = fc_layer(
            input=ipt,
            #size=features_num * 4 * 8 * 8,
            size=features_num * 8 * 4 * 4,
            bias_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.0, name="fc2_b"),
            param_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.02, name="fc2_w"),
            act=ReluActivation())

        convt0 = conv_bn(
            hidden4,
            channels=features_num * 8,
            output_x=4,
            num_filters=features_num * 4,
            imgSize=8,
            stride=2,
            name="convt0" + name,
            param_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.02, name="convt0_w"),
            bias_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.0, name="convt0_b"),
            param_attr_bn=ParamAttr(
                initial_mean=1.0, initial_std=0.02, name="convt0_bn"),
            bn=False,
            trans=True)

        convt1 = conv_bn(
            #hidden4,
            convt0,
            channels=features_num * 4,
            output_x=8,
            num_filters=features_num * 2,
            imgSize=16,
            stride=2,
            name="convt1" + name,
            param_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.02, name="convt1_w"),
            bias_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.0, name="convt1_b"),
            param_attr_bn=ParamAttr(
                initial_mean=1.0, initial_std=0.02, name="convt1_bn"),
            bn=False,
            trans=True)

        convt2 = conv_bn(
            convt1,
            channels=features_num * 2,
            output_x=16,
            num_filters=features_num,
            imgSize=32,
            stride=2,
            name="convt2" + name,
            param_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.02, name="convt2_w"),
            bias_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.0, name="convt2_b"),
            param_attr_bn=ParamAttr(
                initial_mean=1.0, initial_std=0.02, name="convt2_bn"),
            bn=False,
            trans=True)

        convt3 = conv_bn(
            convt2,
            channels=features_num,
            output_x=32,
            num_filters=1,
            imgSize=64,
            stride=2,
            name="convt3" + name,
            param_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.02, name="convt3_w"),
            bias_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.0, name="convt3_b"),
            param_attr_bn=ParamAttr(
                initial_mean=1.0, initial_std=0.02, name="convt3_bn"),
            bn=False,
            trans=True,
            act=ReluActivation())

        return convt3

    recon = deconv(hidden3, "recon")
    future = deconv(hidden2, "pred")

    if is_generating:
        outputs(concat_layer(input=[convt3, trg_image]))
    else:
        with mixed_layer() as entropy:
            entropy += dotmul_operator(
                id, layer_math.log(id + 1e-10), scale=-1.0)
        cost5 = sum_cost(entropy) * invWgt
        #         
        #         id_bn = batch_norm_layer(
        #             id,
        #             bias_attr=bias_attr,
        #             param_attr=param_attr_bn,
        #             use_global_stats=False)
        #         
        #         shape_bn = batch_norm_layer(
        #             shape,
        #             bias_attr=bias_attr,
        #             param_attr=param_attr_bn,
        #             use_global_stats=False)
        #         
        #         pose_bn = batch_norm_layer(
        #             pose,
        #             bias_attr=bias_attr,
        #             param_attr=param_attr_bn,
        #             use_global_stats=False)
        #         
        #         cost4 = sum_cost(out_prod_layer(id_bn, shape_bn)) \
        #                 + sum_cost(out_prod_layer(id_bn, pose_bn)) \
        #                 + sum_cost(out_prod_layer(shape_bn, pose_bn))

        cost3 = classification_cost(input=id, label=lbl, weight=wgt)
        cost1 = regression_cost(input=future, label=trg_image)
        cost2 = regression_cost(
            input=recon,
            label=expand_layer(
                input=first_seq(src_image), expand_as=src_image))

        sum_evaluator(cost1, name="cost1")
        sum_evaluator(cost2, name="cost2")
        sum_evaluator(cost3, name="cost3")
        #         sum_evaluator(cost4, name="cost4")
        sum_evaluator(cost5, name="cost5")
        outputs(cost1, cost2, cost3, cost5)
