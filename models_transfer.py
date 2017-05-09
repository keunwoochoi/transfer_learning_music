# 2017-02-02 Updating for intermediate layer-outputing test
# 2016-06-06 Updating for Keras 1.0 API
import os
import pdb
import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import Convolution1D
from keras.layers import Reshape, Permute
from keras.layers.recurrent import LSTM, GRU
import time
from kapre.time_frequency import Spectrogram, Melspectrogram
from kapre.utils import Normalization2D

SR = 12000


def build_convnet_model(args, last_layer=True, sr=None):
    ''' '''
    # ------------------------------------------------------------------#
    tf = args.tf_type
    normalize = args.normalize
    if normalize in ('no', 'False'):
        normalize = None
    decibel = args.decibel
    model = raw_vgg(args, tf=tf, normalize=normalize, decibel=decibel,
                    last_layer=last_layer, sr=sr)

    model.compile(optimizer=keras.optimizers.Adam(lr=5e-3),
                  loss='binary_crossentropy')
    return model


def raw_vgg(args, input_length=12000 * 29, tf='melgram', normalize=None,
            decibel=False, last_layer=True, sr=None):
    ''' when length = 12000*29 and 512/256 dft/hop,
    melgram size: (n_mels, 1360)
    '''
    assert tf in ('stft', 'melgram')
    assert normalize in (None, False, 'no', 0, 0.0, 'batch', 'data_sample', 'time', 'freq', 'channel')
    assert isinstance(decibel, bool)

    if sr is None:
        sr = SR  # assumes 12000
    conv_until = args.conv_until  # for intermediate layer outputting.
    trainable_kernel = args.trainable_kernel
    model = Sequential()
    if tf == 'stft':
        # decode args
        model.add(Spectrogram(n_dft=512, n_hop=256, power_spectrogram=2.0,
                              trainable_kernel=trainable_kernel,
                              return_decibel_spectrogram=decibel,
                              input_shape=(1, input_length)))
        poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
    elif tf == 'melgram':
        # decode args
        fmin = args.fmin
        fmax = args.fmax
        if fmax == 0.0:
            fmax = sr / 2
        n_mels = args.n_mels
        trainable_fb = args.trainable_fb
        # pdb.set_trace()
        model.add(Melspectrogram(n_dft=512, n_hop=256, power_melgram=2.0,
                                 input_shape=(1, input_length),
                                 trainable_kernel=trainable_kernel,
                                 trainable_fb=trainable_fb,
                                 return_decibel_melgram=decibel,
                                 sr=sr, n_mels=n_mels,
                                 fmin=fmin, fmax=fmax,
                                 name='melgram'))
        if n_mels >= 256:
            poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
        elif n_mels >= 128:
            poolings = [(2, 4), (4, 4), (2, 5), (2, 4), (4, 4)]
        elif n_mels >= 96:
            poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
        elif n_mels >= 72:
            poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (3, 4)]
        elif n_mels >= 64:
            poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (4, 4)]
        elif n_mels >= 48:
            poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (3, 4)]
        elif n_mels >= 32:
            poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (2, 4)]
        elif n_mels >= 24:
            poolings = [(2, 4), (2, 4), (2, 5), (3, 4), (1, 4)]
        elif n_mels >= 18:
            poolings = [(2, 4), (1, 4), (3, 5), (1, 4), (3, 4)]
        elif n_mels >= 18:
            poolings = [(2, 4), (1, 4), (3, 5), (1, 4), (3, 4)]
        elif n_mels >= 16:
            poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (1, 4)]
        elif n_mels >= 12:
            poolings = [(2, 4), (1, 4), (2, 5), (3, 4), (1, 4)]
        elif n_mels >= 8:
            poolings = [(2, 4), (1, 4), (2, 5), (2, 4), (1, 4)]
        elif n_mels >= 6:
            poolings = [(2, 4), (1, 4), (3, 5), (1, 4), (1, 4)]
        elif n_mels >= 4:
            poolings = [(2, 4), (1, 4), (2, 5), (1, 4), (1, 4)]
        elif n_mels >= 2:
            poolings = [(2, 4), (1, 4), (1, 5), (1, 4), (1, 4)]
        else:  # n_mels == 1
            poolings = [(1, 4), (1, 4), (1, 5), (1, 4), (1, 4)]

    else:
        raise RuntimeError('choose between stft or melgram, not %s' % str(tf))
    if normalize in ('batch', 'data_sample', 'time', 'freq', 'channel'):
        # pdb.set_trace()
        model.add(Normalization2D(normalize))
    args = [5,
            [32, 32, 32, 32, 32],
            1.0,
            [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
            poolings,
            0.0, model.output_shape[1:]]
    model.add(get_convBNeluMPdrop(*args, num_nin_layers=1, conv_until=conv_until))
    if conv_until != 4:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())
    if last_layer:
        model.add(Dense(50, activation='sigmoid'))
    return model


def get_convBNeluMPdrop(num_conv_layers, nums_feat_maps, feat_scale_factor,
                        conv_sizes, pool_sizes, dropout_conv, input_shape,
                        num_nin_layers=1, conv_until=None):
    # [Convolutional Layers]
    model = Sequential(name='ConvBNEluDr')
    input_shape_specified = False
    if conv_until is None:
        conv_until = num_conv_layers  # end-inclusive.
    for conv_idx in xrange(num_conv_layers):
        # add conv layer
        n_feat_here = int(nums_feat_maps[conv_idx] * feat_scale_factor)
        for _ in xrange(num_nin_layers):
            if not input_shape_specified:
                model.add(Convolution2D(n_feat_here, conv_sizes[conv_idx][0], conv_sizes[conv_idx][1],
                                        input_shape=input_shape,
                                        border_mode='same',
                                        init='he_normal'))
                input_shape_specified = True
            else:
                model.add(Convolution2D(n_feat_here, conv_sizes[conv_idx][0], conv_sizes[conv_idx][1],
                                        border_mode='same',
                                        init='he_normal'))
            # add BN, Activation, pooling, and dropout
            model.add(BatchNormalization(axis=1, mode=2))
            model.add(keras.layers.advanced_activations.ELU(alpha=1.0))  # TODO: select activation

        model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
        if not dropout_conv == 0.0:
            model.add(Dropout(dropout_conv))
        if conv_idx == conv_until:
            break
    # model.summary()
    return model
