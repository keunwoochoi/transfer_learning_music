import sys
from argparse import Namespace
import librosa
from keras import backend as K
from models_transfer import build_convnet_model
import numpy as np
import keras
import kapre

SR = 12000  # [Hz]
LEN_SRC = 29.  # [second]
ref_n_src = 12000 * 29

if keras.__version__[0] != '1':
    raise RuntimeError('Keras version should be 1.x, maybe 1.2.2')


def load_model(mid_idx):
    assert 0 <= mid_idx <= 4
    args = Namespace(test=False, data_percent=100, model_name='', tf_type='melgram',
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,
                     n_mels=96, trainable_fb=False, trainable_kernel=False,
                     conv_until=mid_idx)
    model = build_convnet_model(args, last_layer=False)
    model.load_weights('weights_transfer/weights_layer{}_{}.hdf5'.format(mid_idx, K._backend),
                       by_name=True)
    return model


def load_audio(audio_path):
    src, sr = librosa.load(audio_path, sr=SR, duration=LEN_SRC)
    len_src = len(src)
    if len_src < ref_n_src:
        new_src = np.zeros(ref_n_src)
        new_src[:len_src] = src
        return new_src[np.newaxis, np.newaxis, :]
    else:
        return src[np.newaxis, np.newaxis, :ref_n_src]


def main(txt_path, out_path):
    models = [load_model(mid_idx) for mid_idx in range(5)]  # for five models...
    all_features = []
    with open(txt_path) as f_path:
        for line in f_path:
            path = line.rstrip('\n')
            print('Loading/extracting {}...'.format(path))
            src = load_audio(path)
            features = [models[i].predict(src)[0] for i in range(5)]
            all_features.append(np.concatenate(features, axis=0))
    all_features = np.array(all_features, dtype=np.float32)
    print('Saving all features at {}..'.format(out_path))
    np.save(out_path, all_features)
    print('Done. Saved a numpy array size of (%d, %d)' % all_features.shape)


def warning():
    print('-' * 65)
    print('  * Python 2.7-ish')
    print('  * Keras 1.2.2,')
    print('  * Kapre old one (git checkout a3bde3e)')
    print("  * Read README.md. Come on, it's short..")
    print('')
    print('   Usage: ')
    print('$ python easy_feature_extraction.py audio_paths.txt features.npy')
    print ''
    print('    , where audio_path.txt is for paths audio line-by-line')
    print('    and features.npy is the path to store result feature array.')
    print('-' * 65)


if __name__ == '__main__':
    warning()

    txt_path = sys.argv[1]
    out_path = sys.argv[2]
    main(txt_path, out_path)
