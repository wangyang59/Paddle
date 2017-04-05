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


def my_img_conv_group(input,
                      conv_num_filter,
                      pool_size,
                      num_channels=None,
                      conv_padding=1,
                      conv_filter_size=3,
                      conv_act=ReluActivation(),
                      conv_with_batchnorm=False,
                      conv_batchnorm_drop_rate=0,
                      pool_stride=1,
                      pool_type=MaxPooling(),
                      param_name=None):
    """
    Image Convolution Group, Used for vgg net.

    TODO(yuyang18): Complete docs

    :param conv_batchnorm_drop_rate:
    :param input:
    :param conv_num_filter:
    :param pool_size:
    :param num_channels:
    :param conv_padding:
    :param conv_filter_size:
    :param conv_act:
    :param conv_with_batchnorm:
    :param pool_stride:
    :param pool_type:
    :return:
    """
    tmp = input

    # Type checks
    assert isinstance(tmp, LayerOutput)
    assert isinstance(conv_num_filter, list) or isinstance(conv_num_filter,
                                                           tuple)
    for each_num_filter in conv_num_filter:
        assert isinstance(each_num_filter, int)

    assert isinstance(pool_size, int)

    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            return obj

    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    conv_act = __extend_list__(conv_act)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)

    for i in xrange(len(conv_num_filter)):
        extra_kwargs = dict()
        if num_channels is not None:
            extra_kwargs['num_channels'] = num_channels
            num_channels = None
        if conv_with_batchnorm[i]:
            extra_kwargs['act'] = LinearActivation()
        else:
            extra_kwargs['act'] = conv_act[i]

        tmp = img_conv_layer(
            input=tmp,
            padding=conv_padding[i],
            filter_size=conv_filter_size[i],
            num_filters=conv_num_filter[i],
            param_attr=ParamAttr(
                initial_mean=0.0,
                initial_std=0.02,
                name=param_name + str(i) + "_w"),
            bias_attr=ParamAttr(
                initial_mean=0.0,
                initial_std=0.0,
                name=param_name + str(i) + "_b")**extra_kwargs)

        # logger.debug("tmp.num_filters = %d" % tmp.num_filters)

        if conv_with_batchnorm[i]:
            dropout = conv_batchnorm_drop_rate[i]
            param_attr_bn = ParamAttr(
                initial_mean=1.0,
                initial_std=0.02,
                name=param_name + str(i) + "_bn_w")
            bias_attr_bn = ParamAttr(
                initial_mean=0.0,
                initial_std=0.0,
                name=param_name + str(i) + "_bn_b")
            if dropout == 0 or abs(dropout) < 1e-5:  # dropout not set
                tmp = batch_norm_layer(
                    input=tmp,
                    act=conv_act[i],
                    param_attr=param_attr_bn,
                    bias_attr=bias_attr_bn)
            else:
                tmp = batch_norm_layer(
                    input=tmp,
                    act=conv_act[i],
                    layer_attr=ExtraAttr(drop_rate=dropout),
                    param_attr=param_attr_bn,
                    bias_attr=bias_attr_bn)

    return img_pool_layer(
        input=tmp, stride=pool_stride, pool_size=pool_size, pool_type=pool_type)
