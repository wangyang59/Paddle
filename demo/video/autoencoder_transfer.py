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
        module="dataprovider_super",
        obj="process",
        args={"src_path": data_dir})


def conv_lstm_net():
    features_num = 32
    label_size = 10

    src_image = data_layer(name='source_image_seq', size=64 * 64)
    lbl = data_layer(name="label", size=label_size)
    wgt = data_layer(name='weight', size=1)
    inputs(src_image, lbl, wgt)

    loaded_param_attr = ParamAttr(is_static=True)

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
        bias_attr=loaded_param_attr,
        param_attr=loaded_param_attr,
        act=ReluActivation())

    pose = fc_layer(
        input=hidden1,
        #size=features_num * 2,
        size=2,
        bias_attr=loaded_param_attr,
        param_attr=loaded_param_attr,
        act=ReluActivation())

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

    hidden = fc_layer(
        input=concat_layer([shape]),
        name='fc1_t',
        size=features_num * 2,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation(),
        layer_attr=ExtraAttr(drop_rate=0.5))

    prob = fc_layer(
        input=hidden,
        name='fc2_t',
        size=label_size,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=SoftmaxActivation())

    cost = classification_cost(input=id, label=lbl, weight=wgt)
    outputs(cost)
