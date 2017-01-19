from paddle.trainer_config_helpers import *
from util import *

data_dir = './data2/'
define_py_data_sources2(
    train_list=data_dir + 'train.list',
    test_list=data_dir + 'test.list',
    module='mnist_provider',
    obj='process')

# settings(
#     batch_size=128,
#     learning_rate=0.1 / 128.0,
#     learning_method=MomentumOptimizer(0.9),
#     regularization=L2Regularization(0.0005 * 128))

settings(learning_method=AdamOptimizer(), batch_size=64, learning_rate=5e-4)

loaded_param_attr = ParamAttr(is_static=True)
img_size = 28

data_size = img_size * img_size
label_size = 10
img = data_layer(name='pixel', size=data_size)
lbl = data_layer(name="label", size=label_size)
inputs(img, lbl)

param_attr = ParamAttr(initial_mean=0.0, initial_std=0.02)
bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.0)

features_num = 32

conv1 = img_conv_layer(
    img,
    filter_size=4,
    num_filters=features_num,
    name="conv1" + "_conv",
    num_channels=1,
    act=ReluActivation(),
    groups=1,
    stride=2,
    padding=3,
    bias_attr=loaded_param_attr,
    param_attr=loaded_param_attr,
    shared_biases=True,
    layer_attr=None,
    filter_size_y=None,
    stride_y=None,
    padding_y=None,
    trans=False)

conv2 = conv_bn(
    conv1,
    channels=features_num,
    imgSize=16,
    num_filters=features_num * 2,
    output_x=8,
    stride=2,
    name="conv2",
    param_attr=loaded_param_attr,
    bias_attr=loaded_param_attr,
    param_attr_bn=loaded_param_attr,
    bn=False)

conv3 = conv_bn(
    conv2,
    channels=features_num * 2,
    imgSize=8,
    num_filters=features_num * 4,
    output_x=4,
    stride=2,
    name="conv3",
    param_attr=loaded_param_attr,
    bias_attr=loaded_param_attr,
    param_attr_bn=loaded_param_attr,
    bn=False)

pooled = img_pool_layer(
    input=conv3,
    num_channels=features_num * 4,
    stride=1,
    pool_size=4,
    pool_type=MaxPooling())

hidden = fc_layer(
    input=pooled,
    size=features_num,
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
