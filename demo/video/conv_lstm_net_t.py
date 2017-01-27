from paddle.trainer_config_helpers import *
from paddle.trainer_config_helpers.layers import get_output_layer
from util import *

error_clipping = 10.0


def seq_to_image_data(data_dir):
    """
    Predefined seqToseq train data provider for application
    is_generating: whether this config is used for generating
    """
    define_py_data_sources2(
        "./data/train.list",
        "./data/test.list",
        module="dataprovider_t",
        obj="process",
        args={"src_path": data_dir})


def create_conv_lstm_step(img_size,
                          channels,
                          output_size,
                          num_filters,
                          stride,
                          num_inputs,
                          nameIdx,
                          param_attr=None):
    input_size = img_size * img_size * channels
    hidden_size = output_size * output_size * num_filters

    filter_size1, padding1 = cal_filter_padding(img_size, output_size, stride)
    filter_size2, padding2 = cal_filter_padding(output_size, output_size, 1)

    def conv_lstm_step(current_img):
        hidden_mem = memory(name='lstm_h' + nameIdx, size=hidden_size)
        cell_mem = memory(name='lstm_c' + nameIdx, size=hidden_size)

        concat = concat_layer(input=[current_img, hidden_mem])

        with mixed_layer(
                size=hidden_size * 4,
                layer_attr=ExtraAttr(
                    error_clipping_threshold=error_clipping)) as ipt:
            ipt += conv_projection(
                concat,
                filter_size2,
                num_filters * 4,
                num_channels=num_filters * 2,
                stride=1,
                padding=padding2,
                param_attr=param_attr)

        lstm_out = lstm_step_layer(
            ipt,
            cell_mem,
            hidden_size,
            act=TanhActivation(),
            name="lstm_h" + nameIdx,
            gate_act=SigmoidActivation(),
            state_act=TanhActivation(),
            bias_attr=param_attr)

        cell_out = get_output_layer(
            name="lstm_c" + nameIdx, input=lstm_out, arg_name='state')
        return lstm_out, cell_out

    def conv_lstm_step2(current_img, hidden_below):
        hidden_mem = memory(name='lstm_h' + nameIdx, size=hidden_size)
        cell_mem = memory(name='lstm_c' + nameIdx, size=hidden_size)

        concat = concat_layer(input=[current_img, hidden_mem])

        ipt = mixed_layer(
            size=hidden_size * 4,
            input=[
                conv_projection(
                    hidden_below,
                    filter_size1,
                    num_filters * 4,
                    num_channels=channels,
                    stride=stride,
                    padding=padding1,
                    param_attr=param_attr), conv_projection(
                        concat,
                        filter_size2,
                        num_filters * 4,
                        num_channels=num_filters * 2,
                        stride=1,
                        padding=padding2,
                        param_attr=param_attr)
            ],
            layer_attr=ExtraAttr(error_clipping_threshold=error_clipping))

        lstm_out = lstm_step_layer(
            ipt,
            cell_mem,
            hidden_size,
            act=TanhActivation(),
            name="lstm_h" + nameIdx,
            gate_act=SigmoidActivation(),
            state_act=TanhActivation(),
            bias_attr=param_attr)

        cell_out = get_output_layer(
            name="lstm_c" + nameIdx, input=lstm_out, arg_name='state')
        return lstm_out, cell_out

    if num_inputs == 1:
        return conv_lstm_step
    else:
        return conv_lstm_step2


def create_convt_lstm_step(img_size,
                           channels,
                           output_size,
                           num_filters,
                           stride,
                           num_inputs,
                           boot_h,
                           boot_c,
                           nameIdx,
                           param_attr=None):
    #     param_attr = ParamAttr(initial_mean=0.0, initial_std=0.02)
    #     bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.0)

    input_size = output_size * output_size * channels
    hidden_size = img_size * img_size * num_filters

    filter_size1, padding1 = cal_filter_padding(img_size, output_size, stride)
    filter_size2, padding2 = cal_filter_padding(img_size, img_size, 1)

    def convt_lstm_step(current_img):
        hidden_mem = memory(
            name='lstm_h' + nameIdx, size=hidden_size, boot_layer=boot_h)
        cell_mem = memory(
            name='lstm_c' + nameIdx, size=hidden_size, boot_layer=boot_c)

        with mixed_layer(
                size=hidden_size * 4,
                layer_attr=ExtraAttr(
                    error_clipping_threshold=error_clipping)) as ipt:
            ipt += conv_projection(
                hidden_mem,
                filter_size2,
                num_filters * 4,
                num_channels=num_filters,
                stride=1,
                padding=padding2,
                param_attr=param_attr)

        lstm_out = lstm_step_layer(
            ipt,
            cell_mem,
            hidden_size,
            act=TanhActivation(),
            name="lstm_h" + nameIdx,
            gate_act=SigmoidActivation(),
            state_act=TanhActivation(),
            bias_attr=param_attr)

        get_output_layer(
            name="lstm_c" + nameIdx, input=lstm_out, arg_name='state')
        return lstm_out

    def convt_lstm_step2(hidden_below):
        hidden_mem = memory(
            name='lstm_h' + nameIdx, size=hidden_size, boot_layer=boot_h)
        cell_mem = memory(
            name='lstm_c' + nameIdx, size=hidden_size, boot_layer=boot_c)

        ipt = img_conv_layer(
            hidden_below,
            filter_size=filter_size1,
            num_filters=num_filters * 4,
            name="lstm_convt" + nameIdx,
            num_channels=channels,
            act=ReluActivation(),
            groups=1,
            stride=stride,
            padding=padding1,
            bias_attr=param_attr,
            param_attr=param_attr,
            shared_biases=True,
            layer_attr=None,
            filter_size_y=None,
            stride_y=None,
            padding_y=None,
            trans=True)

        ipt = mixed_layer(
            size=hidden_size * 4,
            input=[
                identity_projection(ipt), conv_projection(
                    hidden_mem,
                    filter_size2,
                    num_filters * 4,
                    num_channels=num_filters,
                    stride=1,
                    padding=padding2,
                    param_attr=param_attr)
            ],
            layer_attr=ExtraAttr(error_clipping_threshold=error_clipping))

        lstm_out = lstm_step_layer(
            ipt,
            cell_mem,
            hidden_size,
            act=TanhActivation(),
            name="lstm_h" + nameIdx,
            gate_act=SigmoidActivation(),
            state_act=TanhActivation(),
            bias_attr=param_attr)

        get_output_layer(
            name="lstm_c" + nameIdx, input=lstm_out, arg_name='state')
        return lstm_out

    if num_inputs == 0:
        return convt_lstm_step
    else:
        return convt_lstm_step2


def conv_lstm_net_t():
    features_num = 32

    label_size = 10

    src_image = data_layer(name='source_image_seq', size=64 * 64)
    lbl = data_layer(name='label', size=label_size)
    inputs(src_image, lbl)

    loaded_param_attr = ParamAttr(is_static=True)
    param_attr = ParamAttr(initial_mean=0.0, initial_std=0.02)
    bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.0)

    conv1 = conv_bn(
        src_image,
        channels=1,
        imgSize=64,
        num_filters=features_num,
        output_x=32,
        stride=2,
        name="conv1",
        param_attr=loaded_param_attr,
        bias_attr=loaded_param_attr,
        param_attr_bn=loaded_param_attr,
        bn=False)

    conv2 = conv_bn(
        conv1,
        channels=features_num,
        imgSize=32,
        num_filters=features_num * 2,
        output_x=16,
        stride=2,
        name="conv2",
        param_attr=loaded_param_attr,
        bias_attr=loaded_param_attr,
        param_attr_bn=loaded_param_attr,
        bn=False)

    conv3 = conv_bn(
        conv2,
        channels=features_num * 2,
        imgSize=16,
        num_filters=features_num * 4,
        output_x=8,
        stride=2,
        name="conv3",
        param_attr=loaded_param_attr,
        bias_attr=loaded_param_attr,
        param_attr_bn=loaded_param_attr,
        bn=False)

    conv_lstm_step1 = create_conv_lstm_step(
        img_size=64,
        channels=1,
        output_size=32,
        num_filters=features_num,
        stride=2,
        num_inputs=1,
        nameIdx="1",
        param_attr=loaded_param_attr)
    encoder_layer1, cell_layer1 = recurrent_group(
        name="encoder_1", step=conv_lstm_step1, input=conv1)

    conv_lstm_step2 = create_conv_lstm_step(
        img_size=32,
        channels=features_num,
        output_size=16,
        num_filters=features_num * 2,
        stride=2,
        num_inputs=1,
        nameIdx="2",
        param_attr=loaded_param_attr)
    encoder_layer2, cell_layer2 = recurrent_group(
        name="encoder_2", step=conv_lstm_step2, input=conv2)

    conv_lstm_step3 = create_conv_lstm_step(
        img_size=16,
        channels=features_num * 2,
        output_size=8,
        num_filters=features_num * 4,
        stride=2,
        num_inputs=1,
        nameIdx="3",
        param_attr=loaded_param_attr)
    encoder_layer3, cell_layer3 = recurrent_group(
        name="encoder_3", step=conv_lstm_step3, input=conv3)

    pooled3c = img_pool_layer(
        input=last_seq(cell_layer3),
        num_channels=features_num * 4,
        stride=1,
        pool_size=8,
        pool_type=MaxPooling())

    pooled3h = img_pool_layer(
        input=last_seq(encoder_layer3),
        num_channels=features_num * 4,
        stride=1,
        pool_size=8,
        pool_type=MaxPooling())

    pooled3i = img_pool_layer(
        input=last_seq(conv3),
        num_channels=features_num * 4,
        stride=1,
        pool_size=8,
        pool_type=MaxPooling())

    pooled2c = img_pool_layer(
        input=last_seq(cell_layer2),
        num_channels=features_num * 2,
        stride=8,
        pool_size=8,
        pool_type=MaxPooling())

    pooled2h = img_pool_layer(
        input=last_seq(encoder_layer2),
        num_channels=features_num * 2,
        stride=8,
        pool_size=8,
        pool_type=MaxPooling())

    pooled2i = img_pool_layer(
        input=last_seq(conv2),
        num_channels=features_num * 2,
        stride=8,
        pool_size=8,
        pool_type=MaxPooling())

    pooled1c = img_pool_layer(
        input=last_seq(cell_layer1),
        num_channels=features_num,
        stride=8,
        pool_size=8,
        pool_type=MaxPooling())

    pooled1h = img_pool_layer(
        input=last_seq(encoder_layer1),
        num_channels=features_num,
        stride=8,
        pool_size=8,
        pool_type=MaxPooling())

    pooled1i = img_pool_layer(
        input=last_seq(conv1),
        num_channels=features_num,
        stride=8,
        pool_size=8,
        pool_type=MaxPooling())

    hidden = fc_layer(
        input=concat_layer([
            pooled3c, pooled3h, pooled3i, pooled2c, pooled2h, pooled2i,
            pooled1c, pooled1h, pooled1i
        ]),
        size=features_num * 2,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    prob = fc_layer(
        input=hidden,
        size=label_size,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=SoftmaxActivation())

    cost = cross_entropy(input=prob, label=lbl)
    classification_error_evaluator(
        input=prob, label=lbl, name='classification_error')
    outputs(cost)
