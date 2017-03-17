""" .py version of 'knn and svm - for many tasks.ipynb'
After using mid_layer features, it's diverging from the ipython notebook - the notebook is outdated now.
"""
import matplotlib
import matplotlib.pyplot as plt

import multiprocessing

plt.style.use('ggplot')

font = {'family': 'consolas',
        'weight': 'light',
        'size': 12}

matplotlib.rc('font', **font)
ggplot_colors = [plt.rcParams['axes.color_cycle'][i] for i in [0, 1, 2, 3, 4, 5, 6]]

import os
import sys
import numpy as np
import librosa
import time
import sklearn
import pdb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from utils_featext import OptionalStandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import logging
import cPickle as cP

PATH_CLS = 'data_classifiers/'  # save classifiers
FOLDER_CSV = 'data_csv/'
FOLDER_FEATS = 'data_feats/'
FOLDER_RESULTS = 'result_transfer/'

try:
    os.mkdir(FOLDER_RESULTS)
except:
    pass


def load_xy_many(taskname, featname='mine', npy_suffix='', logger=None, mid_layer=4):
    """ wrapper for load_xy() for loading and concatenating multiple of them. """
    if featname == 'mfcc':
        x, y = load_xy(taskname, featname, npy_suffix, logger, mid_layer=mid_layer)
    elif featname == 'mine':
        for l_idx, mid_layer_num in enumerate(mid_layer):
            if l_idx == 0:
                x, y = load_xy(taskname, featname, npy_suffix, logger, mid_layer=mid_layer_num)
            else:
                x_new, _ = load_xy(taskname, featname, npy_suffix, logger, mid_layer=mid_layer_num)
                x = np.concatenate((x, x_new), axis=1)
    elif featname == 'mfcc+12345':
        x, _ = load_xy_many(taskname, 'mfcc', npy_suffix, logger, mid_layer)
        x_12345, y = load_xy_many(taskname, 'mine', npy_suffix, logger, [0, 1, 2, 3, 4])
        x = np.concatenate((x, x_12345), axis=1)
    return x, y


def load_xy(task_name, feat_name='mine', npy_suffix='', logger=None, mid_layer=4):
    """

    :param task_name:
    :param feat_name:
    :param npy_suffix:
    :param logger:
    :param mid_layer: ignired if 'mfcc' is used
    :return:
    """
    assert task_name in ('ballroom_extended', 'gtzan_genre', 'gtzan_speechmusic',
                         'emoMusic_a', 'emoMusic_v', 'jamendo_vd', 'urbansound')
    # logger.info('load_xy({}, {}, mid_layer: {}, npy_suffix: {})...'.format(task_name, feat_name, mid_layer, npy_suffix))

    # X
    csv_filename = '{}.csv'.format(task_name)
    if feat_name == 'mine':
        if task_name.startswith('emoMusic'):
            if mid_layer == 4:  # For the last layer, use Max-Pooled one
                npy_filename = '{}{}.npy'.format('emoMusic', npy_suffix)
            else:  # For the others, use Average-Pooled ones
                npy_filename = '{}_layer_{}{}.npy'.format('emoMusic', mid_layer, npy_suffix)
        else:
            if mid_layer == 4:
                npy_filename = '{}{}.npy'.format(task_name, npy_suffix)
            else:
                npy_filename = '{}_layer_{}{}.npy'.format(task_name, mid_layer, npy_suffix)
    elif feat_name == 'mfcc':
        if task_name.startswith('emoMusic'):
            npy_filename = '{}_mfcc.npy'.format('emoMusic')
        else:
            npy_filename = '{}_mfcc.npy'.format(task_name)

    x = np.load(os.path.join(FOLDER_FEATS, npy_filename))
    # Y
    if task_name == 'emoMusic_v':
        csv_filename = '{}.csv'.format('emoMusic')
        df = pd.DataFrame.from_csv(os.path.join(FOLDER_CSV, csv_filename))
        y = df['label_valence']
    elif task_name == 'emoMusic_a':
        csv_filename = '{}.csv'.format('emoMusic')
        df = pd.DataFrame.from_csv(os.path.join(FOLDER_CSV, csv_filename))
        y = df['label_arousal']
    else:
        y = pd.DataFrame.from_csv(os.path.join(FOLDER_CSV, csv_filename))['label']
    return x, y


def save_result(featname, taskname, classifiername, score):
    """featname: string, taskname:string, score:float"""
    filename = 'T_{}_F_{}_CL_{}.npy'.format(taskname, featname, classifiername)
    np.save(os.path.join(FOLDER_RESULTS, filename), score)


def cross_validate(featnames, tasknames, cvs, classifiers, gps, logger, n_jobs, npy_suffix='', mid_layer=4):
    '''featnames: list of string, ['mine', 'mfcc']

    - tasknames = list of stringm ['ballroom_extended', 'gtzan_genre', 'gtzan_speechmusic',
                                   'emoMusic', 'jamendo_vc', 'urbansound']
    - cvs: list of cv, 10 for rest, split arrays for urbansound and jamendo_vd

    - classifier: list of classifier class, e.g [KNeighborsClassifier, SVC]

    - gps: list of gp, e.g. [{"n_neighbors":[1, 2, 8, 12, 16]}, {"C":[0.1, 8.0], "kernel":['linear', 'rbf']}]

    - mid_layer: scalar, or list of scalar .

    '''

    np.random.seed(1209)

    if not isinstance(mid_layer, list):
        mid_layer = [mid_layer]
    logger.info('')
    logger.info('--- Cross-validation started for {} ---'.format(''.join([str(i) for i in mid_layer])))
    for featname in featnames:
        logger.info(' * feat_name: {} ---'.format(featname))
        for classifier, gp in zip(classifiers, gps):
            clname = classifier.__name__
            logger.info('   - classifier: {} ---'.format(clname))
            for taskname, cv in zip(tasknames, cvs):
                logger.info('     . task: {} ---'.format(taskname))
                model_filename = 'clf_{}_{}_{}.cP'.format(featname, taskname, clname)
                x, y = load_xy_many(taskname, featname, npy_suffix, logger, mid_layer=mid_layer)
                estimators = [('stdd', OptionalStandardScaler()), ('clf', classifier())]
                pipe = Pipeline(estimators)

                if isinstance(gp, dict):  # k-nn or svm with single kernel
                    params = {'stdd__on': [True, False]}
                    params.update({'clf__' + key: value for (key, value) in gp.iteritems()})
                elif isinstance(gp, list):  # svm: grid param can be a list of dictionaries
                    params = []
                    for dct in gp:  # should be dict of list for e.g. svm
                        sub_params = {'stdd__on': [True, False]}
                        sub_params.update({'clf__' + key: value for (key, value) in dct.iteritems()})
                        params.append(sub_params)

                clf = GridSearchCV(pipe, params, cv=cv, n_jobs=n_jobs, pre_dispatch='8*n_jobs').fit(x, y)
                logger.info('     . best score {}'.format(clf.best_score_))
                logger.info(clf.best_params_)
                print('best score of {}, {}, {}: {}'.format(featname,
                                                            taskname,
                                                            clname,
                                                            clf.best_score_))
                print(clf.best_params_)
                cP.dump(clf, open(os.path.join(PATH_CLS, model_filename), 'w'))
                featname_midlayer = '{}_{}'.format(featname, ''.join([str(i) for i in mid_layer]))
                save_result(featname_midlayer, taskname, clname, clf.best_score_)


def get_cv_jamendo():
    task_name = 'jamendo_vd'
    csv_filename = '{}.csv'.format(task_name)
    df = pd.DataFrame.from_csv(os.path.join(FOLDER_CSV, csv_filename))
    splits = 0 * np.array([df['category'] == 'train']).astype(int) \
             + 0 * np.array([df['category'] == 'valid']).astype(int) \
             + 1 * np.array([df['category'] == 'test']).astype(int)
    # PredefinedSplit(df['category'])
    train_idxs = np.where(np.any(splits == 0, axis=0))[0]
    test_idxs = np.where(np.any(splits == 1, axis=0))[0]
    cv_iter = [(train_idxs, test_idxs)]
    return cv_iter


def get_cv_urbansound():
    task_name = 'urbansound'
    csv_filename = '{}.csv'.format(task_name)
    df = pd.DataFrame.from_csv(os.path.join(FOLDER_CSV, csv_filename))
    ps = PredefinedSplit(df['fold'])
    return ps


def get_logger(task_idx, system_name='', memo=''):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler('feature-transfer_task_{}_{}.log'.format(task_idx, system_name))
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('-' * 50)
    logger.info('.' * 50)
    logger.info('memo: {}'.format(memo))
    logger.info('.' * 50)
    logger.info('-' * 50)
    return logger


def do_task_svm(task_idx, logger, n_jobs, which_emo='all'):
    """task_idx is 1 based, NOT zero-based.

    """
    assert 1 <= task_idx <= 6

    tasknames_cl = ['ballroom_extended', 'gtzan_genre', 'gtzan_speechmusic',
                    'jamendo_vd', 'urbansound']
    cvs_cl = [10, 10, 10, get_cv_jamendo(), get_cv_urbansound()]

    is_classification = True
    if task_idx <= 3:
        tasknames = tasknames_cl[task_idx - 1: task_idx]
        cvs = cvs_cl[task_idx - 1: task_idx]
    elif task_idx == 4:  # Regression task.
        is_classification = False
        if which_emo == 'all':
            tasknames = ['emoMusic_a', 'emoMusic_v']
            cvs = [10]
        elif which_emo == 'a':
            tasknames = ['emoMusic_a']
            cvs = [10]
        elif which_emo == 'v':
            tasknames = ['emoMusic_v']
            cvs = [10, 10]
    elif task_idx == 5:
        tasknames = tasknames_cl[3:4]
        cvs = cvs_cl[3:4]
    elif task_idx == 6:
        tasknames = tasknames_cl[4:5]
        cvs = cvs_cl[4:5]

    gps = [[{"C": [0.1, 2.0, 8.0, 32.0], "kernel": ['rbf'],
             "gamma": [0.5 ** i for i in [3, 5, 7, 9, 11, 13]] + ['auto']},
            {"C": [0.1, 2.0, 8.0, 32.0], "kernel": ['linear']}
            ]]
    if is_classification:
        classifiers = [SVC]
    else:
        classifiers = [SVR]

    # FOR MFCC+12345 test,....
    one_layers = [[i] for i in range(5)]
    two_layers = [[i, j] for i in range(5) for j in range(i + 1, 5)]
    three_layers = [[i, j, k] for i in range(5) for j in range(i + 1, 5) for k in range(j + 1, 5)]
    four_layers = [range(4), range(1, 5)]
    five_layers = [range(5)]

    # all_layers = five_layers + four_layers + three_layers + two_layers + one_layers  # 1, 2, 10, 10, 5
    for mid_layer in all_layers:
        cross_validate(['mine'], tasknames, cvs, classifiers, gps, logger, n_jobs, mid_layer=mid_layer)
    # MFCC
    cross_validate(['mfcc'], tasknames, cvs, classifiers, gps, logger, n_jobs, mid_layer=None)
    
    cross_validate(['mfcc+12345'], tasknames, cvs, classifiers, gps, logger, n_jobs, mid_layer=None)

def main_all():
    n_cpu = multiprocessing.cpu_count()
    n_jobs = int(n_cpu * 0.8)
    
    task_idxs = range(1, 7)
    for task_idx in task_idxs:
        logger = get_logger(task_idx)
        do_task_svm(task_idx, logger, n_jobs)

if __name__ == '__main__':

    main_all()

