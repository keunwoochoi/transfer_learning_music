# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import multiprocessing
import os
import joblib
import librosa
import numpy as np

import keras
import kapre
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D as GAP2D
from keras.layers import concatenate as concat
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from utils_featext import OptionalStandardScaler

SR = 12000
DURATION_SEC = 29
TRANSFER_MODEL_PATH = "keras2_model/model_best.hdf5"
SVM_PATH = "keras2_model/svm_{version}.joblib"
GTZAN_GENRE_DIR = os.path.expanduser("~/mir_datasets/GTZAN-Genre/gtzan_genre/genres")
GTZAN_GENRE_CSV = "data_csv/gtzan_genre.csv"
KERAS2_FEATURES_PATH = "data_feats/gtzan_genre_keras2.npy"
TRAIN_SPLIT = "data_csv/keras2_features_gtzan_genre_train.txt"
TEST_SPLIT = "data_csv/keras2_features_gtzan_genre_test.txt"


K.set_image_data_format("channels_last")


def load_models():
    return {
        "transfer_model": load_transfer_model(),
        "svm": joblib.load(SVM_PATH.format(version="keras2")),
    }


def predict(models, audio_path):
    """
    Returns dict of {genre: probability}
    """
    transfer_model = models["transfer_model"]
    svm = models["svm"]
    feats = get_features(transfer_model, audio_path)
    probs = svm.predict_proba(feats)[0]
    return dict(zip(svm.classes_, probs))


def load_transfer_model():
    model = keras.models.load_model(
        TRANSFER_MODEL_PATH,
        custom_objects={
            "Melspectrogram": kapre.time_frequency.Melspectrogram,
            "Normalization2D": kapre.utils.Normalization2D,
        },
    )
    feat_layer1 = GAP2D()(model.get_layer("elu_1").output)
    feat_layer2 = GAP2D()(model.get_layer("elu_2").output)
    feat_layer3 = GAP2D()(model.get_layer("elu_3").output)
    feat_layer4 = GAP2D()(model.get_layer("elu_4").output)
    feat_layer5 = GAP2D()(model.get_layer("elu_5").output)

    feat_all = concat([feat_layer1, feat_layer2, feat_layer3, feat_layer4, feat_layer5])

    return Model(inputs=model.input, outputs=feat_all)


def load_audio(path):
    n_samples = DURATION_SEC * SR
    audio, _ = librosa.load(path, sr=SR, duration=DURATION_SEC, mono=True)
    audio = audio[:n_samples]
    result = np.zeros(n_samples)
    result[: len(audio)] = audio
    return result


def get_features(model, path):
    mfcc_feats = get_mfcc_features(path)
    transfer_feats = get_transfer_features(model, path)[0]
    feats = np.concatenate((mfcc_feats, transfer_feats))
    return feats


def get_transfer_features(model, path):
    audio = load_audio(path)
    feats = model.predict(audio.reshape([1, 1, len(audio)]))
    return feats


def get_mfcc_features(path):
    src_zeros = np.zeros(1024)  # min length to have 3-frame mfcc's
    src, _ = librosa.load(path, sr=SR, duration=29.0)  # max len: 29s, can be shorter.
    if len(src) < 1024:
        src_zeros[: len(src)] = src
        src = src_zeros

    mfcc = librosa.feature.mfcc(src, SR, n_mfcc=20)
    dmfcc = mfcc[:, 1:] - mfcc[:, :-1]
    ddmfcc = dmfcc[:, 1:] - dmfcc[:, :-1]
    return np.concatenate(
        (
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(dmfcc, axis=1),
            np.std(dmfcc, axis=1),
            np.mean(ddmfcc, axis=1),
            np.std(ddmfcc, axis=1),
        ),
        axis=0,
    )


def train_test_split():
    train_track_ids = []
    test_track_ids = []

    with open(TRAIN_SPLIT) as f:
        train_track_ids = [line.strip() for line in f if line]
    with open(TEST_SPLIT) as f:
        test_track_ids = [line.strip() for line in f if line]

    return train_track_ids, test_track_ids


def iter_tracks():
    with open(GTZAN_GENRE_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, (index, genre, path, _) in enumerate(reader):
            index = int(index)
            assert i == index  # make sure the csv is sorted
            track_id = os.path.basename(path).replace(".au", "")
            yield {"index": index, "id": track_id, "genre": genre}


def get_audio_path(track_id):
    genre = track_id.split(".")[0]
    return os.path.join(
        # assuming the dataset has been downloaded by mirdata,
        # but not including mirdata because of too many dependencies
        GTZAN_GENRE_DIR,
        genre,
        track_id + ".wav",
    )


def get_feature_matrix_and_labels_keras2():
    model = load_transfer_model()

    if os.path.exists(KERAS2_FEATURES_PATH):
        features = np.load(KERAS2_FEATURES_PATH)
        labels = [track["genre"] for track in iter_tracks()]
        assert len(features) == len(labels)
        return features, labels

    cache_dir = "keras2-feature-cache"
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    n_examples = 1000
    features = [None] * n_examples
    labels = [None] * n_examples

    for track in iter_tracks():
        index = track["index"]
        track_id = track["id"]
        genre = track["genre"]

        cache_path = os.path.join(cache_dir, track_id + ".npy")
        if os.path.exists(cache_path):
            feat = np.load(cache_path)
        else:
            audio_path = get_audio_path(track_id)
            feat = get_features(model, audio_path)
            np.save(cache_path, feat)

        features[index] = feat
        labels[index] = genre

    features = np.array(features)
    np.save(KERAS2_FEATURES_PATH, features)

    return features, labels


def get_feature_matrix_and_labels_pretrained():
    features = np.hstack(
        [
            np.load("data_feats/gtzan_genre_{}.npy".format(x))
            for x in ["mfcc", "layer_0", "layer_1", "layer_2", "layer_3", "layer_4"]
        ]
    )
    labels = [track["genre"] for track in iter_tracks()]
    assert len(features) == len(labels)
    return features, labels


def data_for_ids(features, labels, track_ids):
    selected_features = []
    selected_labels = []
    for track in iter_tracks():
        if track["id"] in track_ids:
            index = track["index"]
            selected_features.append(features[index])
            selected_labels.append(labels[index])

    return np.array(selected_features), selected_labels


def grid_search_svm():
    grid_params = [
        {
            "C": [0.1, 2.0, 4.0, 8.0, 16.0, 32.0],
            "kernel": ["rbf"],
            "gamma": [0.5 ** i for i in [3, 7, 9, 13, 17]] + ["auto"],
            "probability": [True],
        },
        {"C": [0.1, 2.0, 8.0, 32.0], "kernel": ["linear"], "probability": [True],},
    ]
    n_jobs = multiprocessing.cpu_count()

    estimators = [("stdd", OptionalStandardScaler()), ("clf", SVC())]
    pipe = Pipeline(estimators)

    params = []
    for param_dict in grid_params:
        sub_params = {"stdd__on": [True, False]}
        sub_params.update({"clf__" + key: value for (key, value) in param_dict.items()})
        params.append(sub_params)

    clf = GridSearchCV(
        pipe,
        params,
        cv=10,
        n_jobs=n_jobs,
        pre_dispatch="8*n_jobs",
        scoring="accuracy",
        verbose=10,
    )
    return clf


def train_svc(version):
    if version == "pretrained":
        features, labels = get_feature_matrix_and_labels_pretrained()
    else:
        features, labels = get_feature_matrix_and_labels_keras2()

    train_track_ids, test_track_ids = train_test_split()
    train_features, train_labels = data_for_ids(features, labels, train_track_ids)
    test_features, test_labels = data_for_ids(features, labels, test_track_ids)

    print("training svm")
    clf = grid_search_svm()

    clf.fit(train_features, train_labels)
    print("     . best score {}".format(clf.best_score_))
    print(clf.best_params_)

    train_score = clf.score(train_features, train_labels)
    test_score = clf.score(test_features, test_labels)

    print(train_score, test_score)

    joblib.dump(clf.best_estimator_, SVM_PATH.format(version=version), compress=9)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features and train SVM for GTZAN genre recognition."
    )
    parser.add_argument("--version", choices=["pretrained", "keras2"], required=True)
    args = parser.parse_args()

    train_svc(args.version)


if __name__ == "__main__":
    main()
