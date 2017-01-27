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
        module="dataprovider2",
        obj="process",
        args={"src_path": data_dir})


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
            ],
            layer_attr=ExtraAttr(error_clipping_threshold=error_clipping))

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
            ],
            layer_attr=ExtraAttr(error_clipping_threshold=error_clipping))

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
    features_num = 32

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

    conv_lstm_step3 = create_conv_lstm_step(
        img_size=16,
        channels=features_num * 2,
        output_size=8,
        num_filters=features_num * 4,
        stride=2,
        num_inputs=1,
        nameIdx="3")
    encoder_layer3, cell_layer3 = recurrent_group(
        name="encoder_3", step=conv_lstm_step3, input=conv3)

    convt_lstm_step4 = create_convt_lstm_step(
        output_size=4,
        channels=1,
        img_size=8,
        num_filters=features_num * 4,
        stride=2,
        num_inputs=0,
        boot_h=last_seq(encoder_layer3),
        boot_c=last_seq(cell_layer3),
        nameIdx="4")
    decoder_layer4 = recurrent_group(
        name="decoder_4", step=convt_lstm_step4, input=trg_image)

    #     convt3 = conv_bn(
    #         decoder_layer6,
    #         channels=features_num,
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

    convt1 = conv_bn(
        decoder_layer4,
        channels=features_num * 4,
        output_x=8,
        num_filters=features_num * 2,
        imgSize=16,
        stride=2,
        name="convt1",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False,
        trans=True)

    convt2 = conv_bn(
        convt1,
        channels=features_num * 2,
        output_x=16,
        num_filters=features_num,
        imgSize=32,
        stride=2,
        name="convt2",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False,
        trans=True)

    convt3 = conv_bn(
        convt2,
        channels=features_num,
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

    if is_generating:
        outputs(concat_layer(input=[convt3, trg_image]))
        #outputs(trg_image)
    else:
        cost = regression_cost(input=convt3, label=trg_image)
        outputs(cost)


def gru():
    img_embed_dim = 1024
    encoder_size = 1024
    decoder_size = 1024

    src_image = data_layer(name='source_image_seq', size=64 * 64)
    trg_image = data_layer(name='target_image_seq', size=64 * 64)
    inputs(src_image, trg_image)

    src_embedding = fc_layer(
        input=src_image,
        name="src_embedding",
        size=img_embed_dim,
        bias_attr=ParamAttr(name="_src_embedding_bias"),
        param_attr=ParamAttr(name="_src_embedding_param"),
        act=ReluActivation())

    def gru_decoder(src):
        decoder_mem = memory(name='gru_decoder', size=decoder_size)
        print("name" + decoder_mem.name)

        with mixed_layer(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += full_matrix_projection(input=src)
            decoder_inputs += full_matrix_projection(input=decoder_mem)

        gru_step = gru_step_layer(
            name='gru_decoder',
            input=decoder_inputs,
            output_mem=decoder_mem,
            size=decoder_size)

        return gru_step

    recurrent_group(name="decoder_group", step=gru_decoder, input=src_embedding)
