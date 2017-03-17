import os
import sys
import numpy as np
import keras
import models
# import arg_parser
from argparse import Namespace
import pandas as pd
import librosa
import time
from multiprocessing import Pool
from joblib import Parallel, delayed
from keras import backend as K
from models_transfer import build_convnet_model
from sklearn.preprocessing import StandardScaler

PATH_DATASETS = '/misc/kcgscratch1/ChoGroup/keunwoo/datasets/'
PATH_PROCESSED = '/misc/kcgscratch1/ChoGroup/keunwoo/datasets_processed/'
FOLDER_CSV = 'data_csv/'
FOLDER_FEATS = 'data_feats/'
FOLDER_WEIGHTS = 'weights_transfer/'

SR = 12000  # [Hz]
len_src = 29.  # [second]
N_JOBS = 9
ref_n_src = 12000 * 29
batch_size = 256


class OptionalStandardScaler(StandardScaler):
    def __init__(self, on=False):
        self.on = on  # bool
        if self.on:
            super(OptionalStandardScaler, self).__init__(with_mean=True, with_std=True)
        else:
            super(OptionalStandardScaler, self).__init__(with_mean=False, with_std=False)


def gen_filepaths(df, dataroot=None):
    if dataroot is None:
        dataroot = PATH_DATASETS
    for filepath in df['filepath']:
        yield os.path.join(dataroot, filepath)


def gen_audiofiles(df, batch_size=256, dataroot=None):
    '''gen single audio file src in a batch_size=1 form for keras model.predict_generator
    df: dataframe
    total_size: integer.
    batch_size: integer.
    dataroot: root path for data'''

    ''''''
    pool = Pool(N_JOBS)

    def _multi_loading(pool, paths):
        srcs = pool.map(_load_audio, paths)
        srcs = np.array(srcs)
        try:
            srcs = srcs[:, np.newaxis, :]
        except:
            pdb.set_trace()

        return srcs

    total_size = len(df)
    n_leftover = int(total_size % batch_size)
    leftover = n_leftover != 0
    n_batch = int(total_size / batch_size)
    gen_f = gen_filepaths(df, dataroot=dataroot)
    print('n_batch: {}, n_leftover: {}, all: {}'.format(n_batch, n_leftover, total_size))

    for batch_idx in xrange(n_batch):
        paths = []
        for inbatch_idx in range(batch_size):
            paths.append(gen_f.next())
        print('..yielding {}/{} batch..'.format(batch_idx, n_batch))
        yield _multi_loading(pool, paths)

    if leftover:
        paths = []
        for inbatch_idx in range(n_leftover):
            paths.append(gen_f.next())
        print('..yielding final batch w {} data sample..'.format(len(paths)))
        yield _multi_loading(pool, paths)


def _load_audio(path, zero_pad=False):
    '''return (N,) shape mono audio signal
    if zero_pad, pad zeros.
    Else, repeat and trim.'''
    src, sr = librosa.load(path, sr=SR, duration=len_src * 12000. / float(SR))
    if len(src) >= ref_n_src:
        return src[:ref_n_src]
    else:
        if zero_pad:
            result = np.zeros(ref_n_src)
            result[:len(src)] = src[:ref_n_src]
            return result
        else:
            n_tile = np.ceil(float(ref_n_src) / len(src)).astype('int')
            src = np.tile(src, n_tile)
            return src[:ref_n_src]


def load_model_for_mid(mid_idx):
    assert 0 <= mid_idx <= 4
    args = Namespace(test=False, data_percent=100, model_name='', tf_type='melgram',
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,
                     n_mels=96, trainable_fb=False, trainable_kernel=False,
                     conv_until=mid_idx)
    model = build_convnet_model(args, last_layer=False)
    model.load_weights(os.path.join(FOLDER_WEIGHTS, 'weights_layer{}_{}.hdf5'.format(mid_idx, K._backend)),
                       by_name=True)
    print('----- model {} weights are loaded. (NO ELM!!!) -----'.format(mid_idx))

    return model


def predict(filename, batch_size, model, dataroot=None, npy_suffix=''):
    if dataroot is None:
        dataroot = PATH_DATASETS
    start = time.time()
    csv_filename = '{}.csv'.format(filename)
    npy_filename = '{}{}.npy'.format(filename, npy_suffix)
    df = pd.DataFrame.from_csv(os.path.join(FOLDER_CSV, csv_filename))
    print('{}: Dataframe with size:{}').format(filename, len(df))
    example_path = os.path.join(dataroot, df['filepath'][0])
    print('An example path - does it exists? {}'.format(os.path.exists(example_path)))
    print(df.columns)
    gen_audio = gen_audiofiles(df, batch_size, dataroot)
    feats = model.predict_generator(generator=gen_audio,
                                    val_samples=len(df),
                                    max_q_size=1)
    np.save(os.path.join(FOLDER_FEATS, npy_filename), feats)
    print('DONE! You! uuuuu uu! in {:6.4f} sec'.format(time.time() - start))


# for mfcc
def get_mfcc(filename, dataroot=None):
    start = time.time()
    csv_filename = '{}.csv'.format(filename)
    npy_filename = '{}_mfcc.npy'.format(filename)
    df = pd.DataFrame.from_csv(os.path.join(FOLDER_CSV, csv_filename))
    print('{}: Dataframe with size:{}').format(filename, len(df))
    print(os.path.exists(os.path.join(dataroot, df['filepath'][0])))
    print(df.columns)
    gen_f = gen_filepaths(df, dataroot=dataroot)

    pool = Pool(N_JOBS)
    paths = list(gen_f)
    feats = pool.map(_path_to_mfccs, paths)
    feats = np.array(feats)
    np.save(os.path.join(FOLDER_FEATS, npy_filename), feats)
    print('MFCC is done! in {:6.4f} sec'.format(time.time() - start))
    pool.close()
    pool.join()


def _path_to_mfccs(path):
    src_zeros = np.zeros(1024)  # min length to have 3-frame mfcc's
    src, sr = librosa.load(path, sr=SR, duration=29.)  # max len: 29s, can be shorter.
    if len(src) < 1024:
        src_zeros[:len(src)] = src
        src = src_zeros

    mfcc = librosa.feature.mfcc(src, SR, n_mfcc=20)
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                           np.mean(dmfcc, axis=1), np.std(dmfcc, axis=1),
                           np.mean(ddmfcc, axis=1), np.std(ddmfcc, axis=1))
                          , axis=0)
