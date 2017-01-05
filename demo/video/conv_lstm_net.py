from paddle.trainer_config_helpers import *
from paddle.trainer_config_helpers.layers import get_output_layer


def seq_to_image_data(data_dir):
    """
    Predefined seqToseq train data provider for application
    is_generating: whether this config is used for generating
    """
    define_py_data_sources2(
        "./data/train.list",
        "./data/test.list",
        module="dataprovider2",
        obj="process",
        args={"src_path": data_dir})


def cal_filter_padding(imgSize, output_size, stride):
    # calculate the filter_size and padding size based on the given
    # imgSize and ouput size
    tmp = imgSize - (output_size - 1) * stride
    if tmp < 1 or tmp > 5:
        print(imgSize, output_size, stride)
        raise ValueError("conv input-output dimension does not fit")
    elif tmp == 1:
        filter_size = 5
        padding = 2
    elif tmp <= 3:
        filter_size = tmp + 2
        padding = 1
    else:
        filter_size = tmp
        padding = 0
    return filter_size, padding


def conv_bn(input,
            channels,
            imgSize,
            num_filters,
            output_x,
            stride,
            name,
            param_attr,
            bias_attr,
            param_attr_bn,
            bn,
            trans=False,
            act=ReluActivation()):
    """
    conv_bn is a utility function that constructs a convolution/deconv layer 
    with an optional batch_norm layer

    :param bn: whether to use batch_norm_layer
    :type bn: bool
    :param trans: whether to use conv (False) or deconv (True)
    :type trans: bool
    """

    # calculate the filter_size and padding size based on the given
    # imgSize and ouput size    
    filter_size, padding = cal_filter_padding(imgSize, output_x, stride)

    print(imgSize, output_x, stride, filter_size, padding)

    if trans:
        nameApx = "_convt"
    else:
        nameApx = "_conv"

    if bn:
        conv = img_conv_layer(
            input,
            filter_size=filter_size,
            num_filters=num_filters,
            name=name + nameApx,
            num_channels=channels,
            act=LinearActivation(),
            groups=1,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr,
            param_attr=param_attr,
            shared_biases=True,
            layer_attr=None,
            filter_size_y=None,
            stride_y=None,
            padding_y=None,
            trans=trans)

        conv_bn = batch_norm_layer(
            conv,
            act=act,
            name=name + nameApx + "_bn",
            bias_attr=bias_attr,
            param_attr=param_attr_bn,
            use_global_stats=False)

        return conv_bn
    else:
        conv = img_conv_layer(
            input,
            filter_size=filter_size,
            num_filters=num_filters,
            name=name + nameApx,
            num_channels=channels,
            act=act,
            groups=1,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr,
            param_attr=param_attr,
            shared_biases=True,
            layer_attr=None,
            filter_size_y=None,
            stride_y=None,
            padding_y=None,
            trans=trans)
        return conv


def create_conv_lstm_step(img_size, channels, output_size, num_filters, stride,
                          num_inputs, nameIdx):
    input_size = img_size * img_size * channels
    hidden_size = output_size * output_size * num_filters

    filter_size1, padding1 = cal_filter_padding(img_size, output_size, stride)
    filter_size2, padding2 = cal_filter_padding(output_size, output_size, 1)

    def conv_lstm_step(current_img):
        hidden_mem = memory(name='lstm_h' + nameIdx, size=hidden_size)
        cell_mem = memory(name='lstm_c' + nameIdx, size=hidden_size)

        concat = concat_layer(input=[current_img, hidden_mem])

        with mixed_layer(size=hidden_size * 4) as ipt:
            ipt += conv_projection(
                concat,
                filter_size2,
                num_filters * 4,
                num_channels=num_filters * 2,
                stride=1,
                padding=padding2)

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
                    padding=padding1), conv_projection(
                        concat,
                        filter_size2,
                        num_filters * 4,
                        num_channels=num_filters * 2,
                        stride=1,
                        padding=padding2)
            ])

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

    if num_inputs == 1:
        return conv_lstm_step
    else:
        return conv_lstm_step2


def create_convt_lstm_step(img_size, channels, output_size, num_filters, stride,
                           num_inputs, boot_h, boot_c, nameIdx):
    param_attr = ParamAttr(initial_mean=0.0, initial_std=0.02)
    bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.0)

    input_size = output_size * output_size * channels
    hidden_size = img_size * img_size * num_filters

    filter_size1, padding1 = cal_filter_padding(img_size, output_size, stride)
    filter_size2, padding2 = cal_filter_padding(img_size, img_size, 1)

    def convt_lstm_step(current_img):
        hidden_mem = memory(
            name='lstm_h' + nameIdx, size=hidden_size, boot_layer=boot_h)
        cell_mem = memory(
            name='lstm_c' + nameIdx, size=hidden_size, boot_layer=boot_c)

        with mixed_layer(size=hidden_size * 4) as ipt:
            ipt += conv_projection(
                hidden_mem,
                filter_size2,
                num_filters * 4,
                num_channels=num_filters,
                stride=1,
                padding=padding2)

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
            bias_attr=bias_attr,
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
                    padding=padding2)
            ])

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

    if num_inputs == 0:
        return convt_lstm_step
    else:
        return convt_lstm_step2


def conv_lstm_net(is_generating):
    src_image = data_layer(name='source_image_seq', size=64 * 64)
    trg_image = data_layer(name='target_image_seq', size=64 * 64)
    inputs(src_image, trg_image)

    param_attr = ParamAttr(initial_mean=0.0, initial_std=0.02)
    bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.0)
    param_attr_bn = ParamAttr(initial_mean=1.0, initial_std=0.02)

    conv1 = conv_bn(
        src_image,
        channels=1,
        imgSize=64,
        num_filters=16,
        output_x=32,
        stride=2,
        name="conv1",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=True)

    conv2 = conv_bn(
        conv1,
        channels=16,
        imgSize=32,
        num_filters=32,
        output_x=16,
        stride=2,
        name="conv2",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=True)

    conv3 = conv_bn(
        conv2,
        channels=32,
        imgSize=16,
        num_filters=64,
        output_x=8,
        stride=2,
        name="conv3",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=True)

    conv_lstm_step1 = create_conv_lstm_step(64, 1, 32, 16, 2, 1, "1")
    encoder_layer1, cell_layer1 = recurrent_group(
        name="encoder_1", step=conv_lstm_step1, input=conv1)

    #     conv_lstm_step2 = create_conv_lstm_step(32, 16, 16, 32, 2, 2, "2")
    #     encoder_layer2, cell_layer2 = recurrent_group(
    #         name="encoder_2", step=conv_lstm_step2, input=[conv2, encoder_layer1])
    # 
    #     conv_lstm_step3 = create_conv_lstm_step(16, 32, 8, 64, 2, 2, "3")
    #     encoder_layer3, cell_layer3 = recurrent_group(
    #         name="encoder_3", step=conv_lstm_step3, input=[conv3, encoder_layer2])

    conv_lstm_step2 = create_conv_lstm_step(32, 16, 16, 32, 2, 1, "2")
    encoder_layer2, cell_layer2 = recurrent_group(
        name="encoder_2", step=conv_lstm_step2, input=conv2)

    conv_lstm_step3 = create_conv_lstm_step(16, 32, 8, 64, 2, 1, "3")
    encoder_layer3, cell_layer3 = recurrent_group(
        name="encoder_3", step=conv_lstm_step3, input=conv3)

    convt_lstm_step4 = create_convt_lstm_step(8, 1, 4, 64, 2, 0,
                                              last_seq(encoder_layer3),
                                              last_seq(cell_layer3), "4")
    decoder_layer4 = recurrent_group(
        name="decoder_4", step=convt_lstm_step4, input=trg_image)

    convt_lstm_step5 = create_convt_lstm_step(16, 64, 8, 32, 2, 1,
                                              last_seq(encoder_layer2),
                                              last_seq(cell_layer2), "5")
    decoder_layer5 = recurrent_group(
        name="decoder_5", step=convt_lstm_step5, input=decoder_layer4)

    convt_lstm_step6 = create_convt_lstm_step(32, 32, 16, 16, 2, 1,
                                              last_seq(encoder_layer1),
                                              last_seq(cell_layer1), "6")
    decoder_layer6 = recurrent_group(
        name="decoder_6", step=convt_lstm_step6, input=decoder_layer5)

    convt3 = conv_bn(
        decoder_layer6,
        channels=16,
        output_x=32,
        num_filters=1,
        imgSize=64,
        stride=2,
        name="convt3",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False,
        trans=True,
        act=SigmoidActivation())

    #     convt1 = conv_bn(
    #         last_seq(encoder_layer3),
    #         channels=64,
    #         output_x=8,
    #         num_filters=32,
    #         imgSize=16,
    #         stride=2,
    #         name="convt1",
    #         param_attr=param_attr,
    #         bias_attr=bias_attr,
    #         param_attr_bn=param_attr_bn,
    #         bn=True,
    #         trans=True)
    # 
    #     convt2 = conv_bn(
    #         concat_layer(input=[convt1, last_seq(encoder_layer2)]),
    #         channels=32 * 2,
    #         output_x=16,
    #         num_filters=16,
    #         imgSize=32,
    #         stride=2,
    #         name="convt2",
    #         param_attr=param_attr,
    #         bias_attr=bias_attr,
    #         param_attr_bn=param_attr_bn,
    #         bn=True,
    #         trans=True)
    # 
    #     convt3 = conv_bn(
    #         concat_layer(input=[convt2, last_seq(encoder_layer1)]),
    #         channels=16 * 2,
    #         output_x=32,
    #         num_filters=1,
    #         imgSize=64,
    #         stride=2,
    #         name="convt3",
    #         param_attr=param_attr,
    #         bias_attr=bias_attr,
    #         param_attr_bn=param_attr_bn,
    #         bn=False,
    #         trans=True,
    #         act=SigmoidActivation())

    if is_generating:
        outputs(concat_layer(input=[convt3, trg_image]))
        #outputs(trg_image)
    else:
        cost = regression_cost(input=convt3, label=trg_image)
        outputs(cost)
