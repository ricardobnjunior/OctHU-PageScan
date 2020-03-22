from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D,GlobalMaxPooling2D, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda, UpSampling2D
from keras.layers import DepthwiseConv2D
import numpy as np
import random
from keras_octave_conv import OctaveConv2D, octave_dual


def octBatchNorm(bn_axis,prefix, x, suffix):
    name_1 = prefix+suffix+'_1'
    name_2 = prefix + suffix + '_2'
    x = [BatchNormalization(axis=bn_axis, name=name_1)(x[0]), BatchNormalization(axis=bn_axis, name=name_2)(x[1])]
    return x

def octActivation(x, type_activation, prefix='', suffix=''):
    name_1 = prefix + suffix + '_1'
    name_2 = prefix + suffix + '_2'
    if prefix == '':
        rand_int = random.randint(1, 1000)
        name = type_activation+'_'+str(rand_int)
        x = [Activation(type_activation)(x[0]), Activation(type_activation)(x[1])]
    else:
        x = [Activation(type_activation, name=name_1)(x[0]), Activation(type_activation, name=name_2)(x[1])]

    return x


def octLambda(channel_shuffle, groups, prefix, suffix, x):
    name_1 = prefix + suffix + '_1'
    name_2 = prefix + suffix + '_2'

    x = [ Lambda(channel_shuffle, arguments={'groups': groups}, name=name_1)(x[0]), Lambda(channel_shuffle, arguments={'groups': groups}, name=name_2)(x[1])]
    return x

def octLambda_second_type(lambda_val, name_lambda, offset, ig, x):
    name_lambda_1 = name_lambda+'_1'
    name_lambda_2 = name_lambda + '_2'

    x = [Lambda(lambda x: x ** 2, name=name_lambda_1)(x[0]), Lambda(lambda x: x ** 2, name=name_lambda_2)(x[1])]

    return x

def octPooling(poolingType, pool_size, strides, padding, prefix, suffix, x):

    name_1 = prefix + suffix + '_1'
    name_2 = prefix + suffix + '_2'

    if 'aver' in poolingType:
        x = [AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name_1)(x[0]),
             AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name_2)(x[1])]
    if 'max' in poolingType:
        x = [MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name_1)(x[0]),
             MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name_2)(x[1])]

    return x


def octUpsize(model, size_upped):
    upsampled = [UpSampling2D(size=(size_upped, size_upped), data_format=None, interpolation='bilinear')(model[0]),
    UpSampling2D(size=(size_upped, size_upped), data_format=None, interpolation='bilinear')(model[1])]

    return upsampled