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
        "./videoData/train_h5.list",
        "./videoData/train_h5.list",
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
    image_size = 256
    image_channel = 3

    src_image = data_layer(
        name='source_image_seq', size=image_size * image_size * image_channel)
    trg_image = data_layer(
        name='target_image_seq', size=image_size * image_size * image_channel)

    inputs(src_image, trg_image)

    param_attr = ParamAttr(initial_mean=0.0, initial_std=0.02)
    bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.0)
    param_attr_bn = ParamAttr(initial_mean=1.0, initial_std=0.02)

    conv1 = conv_bn(
        src_image,
        channels=image_channel,
        imgSize=image_size,
        num_filters=features_num,
        output_x=image_size / 2,
        stride=2,
        name="conv1",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    conv2 = conv_bn(
        conv1,
        channels=features_num,
        imgSize=image_size / 2,
        num_filters=features_num * 2,
        output_x=image_size / 4,
        stride=2,
        name="conv2",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    conv3 = conv_bn(
        conv2,
        channels=features_num * 2,
        imgSize=image_size / 4,
        num_filters=features_num * 4,
        output_x=image_size / 8,
        stride=2,
        name="conv3",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    conv4 = conv_bn(
        conv3,
        channels=features_num * 4,
        imgSize=image_size / 8,
        num_filters=features_num * 8,
        output_x=image_size / 16,
        stride=2,
        name="conv4",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    conv5 = conv_bn(
        conv4,
        channels=features_num * 8,
        imgSize=image_size / 16,
        num_filters=features_num * 16,
        output_x=image_size / 32,
        stride=2,
        name="conv5",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    hidden1 = fc_layer(
        input=conv5,
        size=features_num * 2,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    pose = fc_layer(
        input=hidden1,
        size=features_num,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    shape = fc_layer(
        input=hidden1,
        size=features_num,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    shape_pool = pooling_layer(
        input=shape,
        pooling_type=AvgPooling(),
        agg_level=AggregateLevel.EACH_TIMESTEP)

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
        size=features_num,
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
    hidden2 = concat_layer(
        [expand_layer(
            input=shape_pool, expand_as=new_pose), new_pose])
    hidden3 = concat_layer(
        [shape, expand_layer(
            input=first_seq(pose), expand_as=pose)])

    def deconv(ipt, name):

        hidden4 = fc_layer(
            input=ipt,
            size=features_num * 16 * image_size / 32 * image_size / 32,
            bias_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.0, name="fc2_b"),
            param_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.02, name="fc2_w"),
            act=ReluActivation())

        convt0 = conv_bn(
            hidden4,
            channels=features_num * 16,
            output_x=image_size / 32,
            num_filters=features_num * 8,
            imgSize=image_size / 16,
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
            convt0,
            channels=features_num * 8,
            output_x=image_size / 16,
            num_filters=features_num * 4,
            imgSize=image_size / 8,
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
            channels=features_num * 4,
            output_x=image_size / 8,
            num_filters=features_num * 2,
            imgSize=image_size / 4,
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
            channels=features_num * 2,
            output_x=image_size / 4,
            num_filters=features_num,
            imgSize=image_size / 2,
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

        convt4 = conv_bn(
            convt3,
            channels=features_num,
            output_x=image_size / 2,
            num_filters=image_channel,
            imgSize=image_size,
            stride=2,
            name="convt4" + name,
            param_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.02, name="convt4_w"),
            bias_attr=ParamAttr(
                initial_mean=0.0, initial_std=0.0, name="convt4_b"),
            param_attr_bn=ParamAttr(
                initial_mean=1.0, initial_std=0.02, name="convt4_bn"),
            bn=False,
            trans=True,
            act=ReluActivation())

        return convt4

    recon = deconv(hidden3, "recon")
    future = deconv(hidden2, "pred")

    if is_generating:
        outputs(concat_layer(input=[src_image, trg_image]))
    else:
        cost1 = regression_cost(input=future, label=trg_image)
        cost2 = regression_cost(
            input=recon,
            label=expand_layer(
                input=first_seq(src_image), expand_as=src_image))

        sum_evaluator(cost1, name="cost1")
        sum_evaluator(cost2, name="cost2")
        outputs(cost1, cost2)
