from paddle.trainer_config_helpers import *


def cal_filter_padding(imgSize, output_size, stride):
    # calculate the filter_size and padding size based on the given
    # imgSize and ouput size
    tmp = imgSize - (output_size - 1) * stride
    if tmp < 1 or tmp > 5:
        print(imgSize, output_size, stride)
        raise ValueError("conv input-output dimension does not fit")

#     elif tmp == 1:
#         filter_size = 5
#         padding = 2
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
