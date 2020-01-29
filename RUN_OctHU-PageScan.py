import os

from keras.activations import relu
from keras.layers import *
from keras.models import Model, load_model
import glob
import os.path as P
import cv2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
import threading
from keras.optimizers import Adam
import numpy as np
from keras_octave_conv import OctaveConv2D, octave_dual
from residual_functions import *
from octLayers import *

init_channels = 16
train_folder = r"E:\Ricardo\Datasets\0-BASES DE ARTIGOS\CDPhotoDataset\train"
validation_folder = r"E:\Ricardo\Datasets\0-BASES DE ARTIGOS\CDPhotoDataset\validation"
type_save_image = ".png"
type_of_images = "*.png"
__DEF_HEIGHT = 512
__DEF_WIDTH = 512
input_channels = 1
train_steps = 1
valid_steps = 1
qtd_epochs = 10000
train_samples = 10264
valid_samples = 1140
bs = 4
learning_rate = 0.0001
path_to_save_new_model = "checkpoint"
output_refined = "output"


train_fns = sorted(glob.glob(P.join(train_folder, type_of_images)))
train_fns = [k for k in train_fns if '_gt' not in k]

valid_fns = sorted(glob.glob(P.join(validation_folder, type_of_images)))
valid_fns = [k for k in valid_fns if '_gt' not in k]


def generator_batch(fns, bs, validation=False, stroke=True):
    batches = []
    for i in range(0, len(fns), bs):
        batches.append(fns[i: i + bs])

    print("Batching {} batches of size {} each for {} total files".format(len(batches), bs, len(fns)))
    while True:
        for fns in batches:
            imgs_batch = []
            masks_batch = []
            bounding_batch = []
            for fn in fns:
                if input_channels == 1:
                    _img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
                else:
                    _img = cv2.imread(fn)
                if _img is None:
                    print(fn)
                    continue

                _img = cv2.resize(_img, (__DEF_WIDTH, __DEF_HEIGHT), interpolation=cv2.INTER_CUBIC)
                _img = _img.astype('float32')
                if stroke:
                    gt = "gt"+type_save_image
                else:
                    gt = "gt"+type_save_image

                if input_channels == 1:
                    mask = cv2.imread(fn.replace("in"+type_save_image, gt), cv2.IMREAD_GRAYSCALE)
                else:
                    mask = cv2.imread(fn.replace("in" + type_save_image, gt))
                if mask is None:
                    print(fn)
                    continue

                mask = cv2.resize(mask, (__DEF_WIDTH, __DEF_HEIGHT), interpolation=cv2.INTER_CUBIC)
                _img = 1 - (_img.reshape((__DEF_WIDTH, __DEF_HEIGHT, input_channels)) / 255)
                mask = mask.reshape((__DEF_WIDTH, __DEF_HEIGHT, input_channels)) / 255
                mask = mask > 0.3

                mask = mask.astype('float32')
                imgs_batch.append(_img)
                masks_batch.append(mask)

            imgs_batch = np.asarray(imgs_batch).reshape((bs, __DEF_WIDTH, __DEF_HEIGHT, input_channels)).astype('float32')
            masks_batch = np.asarray(masks_batch).reshape((bs, __DEF_WIDTH, __DEF_HEIGHT, input_channels)).astype('float32')

            yield imgs_batch, masks_batch


def dice_coef(y_true, y_pred, smooth=1000.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def octHu_PageScan():

    inputs = Input(shape=(__DEF_HEIGHT, __DEF_WIDTH, input_channels))
    conv_1 = OctaveConv2D(filters=init_channels, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform', name='octave_conv_1')(inputs)
    conv_1 = OctaveConv2D(filters=init_channels, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform', name='octave_conv_2')(conv_1)
    pool_1 = octPooling('max', 2, 2, 'same', 'oct', '/avg_pool', conv_1)

    conv_2 = OctaveConv2D(filters=init_channels*2, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform',name='octave_conv_3')(pool_1)
    conv_2 = OctaveConv2D(filters=init_channels*2, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform',name='octave_conv_4')(conv_2)
    pool_2 = octPooling('max', 2, 2, 'same', 'oct', '/avg_pool_2', conv_2)

    conv_3 = OctaveConv2D(filters=init_channels*4, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform',name='octave_conv_5')(pool_2)
    conv_3 = OctaveConv2D(filters=init_channels*4, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform',name='octave_conv_6')(conv_3)
    pool_3 = octPooling('max', 2, 2, 'same', 'oct', '/avg_pool_3', conv_3)

    conv_4 = OctaveConv2D(filters=init_channels*8, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform',name='octave_conv_7')(pool_3)
    conv_4 = OctaveConv2D(filters=init_channels*8, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform',name='octave_conv_8')(conv_4)
    pool_4 = octPooling('max', 2, 2, 'same', 'oct', '/avg_pool_4', conv_4)

    conv_5 = OctaveConv2D(filters=init_channels*16, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform',name='octave_conv_9')(pool_4)
    conv_5 = OctaveConv2D(filters=init_channels*16, kernel_size=3, ratio_out=0.5, activation='relu', kernel_initializer='he_uniform',name='octave_conv_10')(conv_5)

    up_6 = octUpsize(conv_5, 2)
    up_6 = [Concatenate()([up_6[0], conv_4[0]]), Concatenate()([up_6[1], conv_4[1]])]
    conv_6 = OctaveConv2D(filters=init_channels*8, kernel_size=3, ratio_out=0.7, activation='relu', kernel_initializer='he_uniform',name='octave_conv_11')(up_6)
    conv_6 = OctaveConv2D(filters=init_channels*8, kernel_size=3, ratio_out=0.7, activation='relu', kernel_initializer='he_uniform',name='octave_conv_12')(conv_6)

    up_7 = octUpsize(conv_6, 2)
    up_7 = [Concatenate()([up_7[0], conv_3[0]]), Concatenate()([up_7[1], conv_3[1]])]
    conv_7 = OctaveConv2D(filters=init_channels*4, kernel_size=3, ratio_out=0.7, activation='relu', kernel_initializer='he_uniform',name='octave_conv_13')(up_7)
    conv_7 = OctaveConv2D(filters=init_channels*4, kernel_size=3, ratio_out=0.7, activation='relu', kernel_initializer='he_uniform',name='octave_conv_14')(conv_7)

    up_8 = octUpsize(conv_7, 2)
    up_8 = [Concatenate()([up_8[0], conv_2[0]]), Concatenate()([up_8[1], conv_2[1]])]
    conv_8 = OctaveConv2D(filters=init_channels*2, kernel_size=3, ratio_out=0.7, activation='relu', kernel_initializer='he_uniform',name='octave_conv_15')(up_8)
    conv_8 = OctaveConv2D(filters=init_channels*2, kernel_size=3, ratio_out=0.7, activation='relu', kernel_initializer='he_uniform',name='octave_conv_16')(conv_8)

    up_9 = octUpsize(conv_8, 2)
    up_9 = [Concatenate()([up_9[0], conv_1[0]]), Concatenate()([up_9[1], conv_1[1]])]
    conv_9 = OctaveConv2D(filters=init_channels, kernel_size=3, ratio_out=0.7, activation='relu', kernel_initializer='he_uniform',name='octave_conv_17')(up_9)
    conv_9 = OctaveConv2D(filters=init_channels, kernel_size=3, ratio_out=0.7, activation='relu', kernel_initializer='he_uniform',name='octave_conv_18')(conv_9)

    conv_10 = OctaveConv2D(filters=input_channels, ratio_out=0, kernel_size=1, activation='sigmoid', kernel_initializer='he_uniform',name='output')(conv_9)
    
    model = Model(inputs=inputs, outputs=conv_10)
    model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()

    return model

def main(train_steps, valid_steps, train_samples, valid_samples, bs, train_fns, valid_fns):
    if train_samples:
        train_fns = train_fns[:int(train_samples)]

    if valid_samples:
        valid_fns = valid_fns[:int(valid_samples)]
    train_samples = len(train_fns) - (len(train_fns) % bs)
    valid_samples = len(valid_fns) - (len(valid_fns) % bs)

    train_fns = train_fns[:train_samples]
    valid_fns = valid_fns[:valid_samples]

    np.random.seed(0)
    np.random.shuffle(train_fns)
    np.random.shuffle(valid_fns)
    np.random.seed()

    callbacks = []
    monitor = 'val_loss'
    monitor_mode = 'min'

    model = octHu_PageScan()
    model.summary()
    checkpoint_model_best = ModelCheckpoint(path_to_save_new_model + '/%s.hdf5' % '0_best_model',
                                            monitor=monitor, save_best_only=True, verbose=1, mode=monitor_mode)

    check_before_epochs = ModelCheckpoint(path_to_save_new_model + '/model_{epoch:05d}.h5',
                                          monitor=monitor, period=1000, verbose=1, mode=monitor_mode)

    callbacks.append(EarlyStopping(
        monitor=monitor, patience=30, verbose=1, mode=monitor_mode,
    ))

    exitlog = CSVLogger('training-resnet.txt')

    train_gen = generator_batch(train_fns, bs=bs, stroke=False)
    valid_gen = generator_batch(valid_fns, bs=bs, validation=True)

    class SaveImageCallback(Callback):
        def __init__(self, stroke=False):
            super(SaveImageCallback, self).__init__()
            self.lock = threading.Lock()

        def on_epoch_end(self, epoch, logs={}):
            self.lock = threading.Lock()
            with self.lock:
                data, gt = next(train_gen)
                mask = self.model.predict_on_batch(data)
                for i in range(mask.shape[0]):
                    cv2.imwrite(output_refined + '/%d-%d-0_mask.png' % (epoch, i), mask[i, :, :, 0] * 255)
                    cv2.imwrite(output_refined + '/%d-%d-1_gt.png' % (epoch, i),gt[i, :, :, 0] * 255)
                    cv2.imwrite(output_refined + '/%d-%d-2_data.png' % (epoch, i), data[i, :, :, 0] * 255)

    save_net = SaveImageCallback()

    model.fit_generator(
        generator=train_gen, steps_per_epoch=train_steps,
        epochs=qtd_epochs,
        verbose=1,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=[save_net, checkpoint_model_best, check_before_epochs, exitlog],
        use_multiprocessing=True,
        workers=0
    )


main(train_steps, valid_steps, train_samples, valid_samples, bs, train_fns, valid_fns)