import os

from keras.layers import *
from keras.models import Model
import glob
import os.path as P
import cv2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
import threading
from keras.optimizers import Adam
import numpy as np
from keras_octave_conv import OctaveConv2D, octave_dual


cardinality = 1



def oct_BatchNormalization(model):
    return [BatchNormalization()(model[0]), BatchNormalization()(model[1])]

def add_common_layers(y):
    y = [BatchNormalization()(y[0]), BatchNormalization()(y[1])]
    #y = BatchNormalization()(y)
    #y = LeakyReLU()(y)
    y = [LeakyReLU()(y[0]), LeakyReLU()(y[1])]

    return y

def grouped_convolution(y, nb_channels, _strides):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        #return Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        return OctaveConv2D(nb_channels, kernel_size=(3, 3), strides=_strides, ratio_out=0.5)(y)

    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        #groups.append(Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
        groups.append(OctaveConv2D(_d, kernel_size=(3, 3), strides=_strides, ratio_out=0.5)(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = concatenate(groups)

    return y

def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    #y = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = OctaveConv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), ratio_out=0.5)(y)
    y = add_common_layers(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = add_common_layers(y)

    #y = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = OctaveConv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), ratio_out=0.5)(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = [BatchNormalization()(y[0]), BatchNormalization()(y[1])]


    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        #shortcut = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = OctaveConv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, ratio_out=0.5)(shortcut)
        #shortcut = oct_BatchNormalization()(shortcut)
        shortcut = [BatchNormalization()(shortcut[0]), BatchNormalization()(shortcut[1])]


    y = [add([shortcut[0], y[0]]), add([shortcut[1], y[1]])]

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = [LeakyReLU()(y[0]), LeakyReLU()(y[1])]

    return y
