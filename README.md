# transfer_learning_music

Repo for paper ["Transfer learning for music classification and regression tasks"](https://arxiv.org/abs/1703.09179) by Keunwoo Choi et al.

![diagram](https://github.com/keunwoochoi/transfer_learning_music/blob/master/diagram.png "diagram")
![results](https://github.com/keunwoochoi/transfer_learning_music/blob/master/results.png "results")

# Mode 1/2. To use the pre-trained convnet feature extractor

For your own music/audio-related work.

## Prerequisites (Same as mode 2 except datasets)
  - [Theano](http://deeplearning.net/software/theano/index.html). I used version 0.9.0 but should work with some similar versions.
  - [Keras 1.2.2 (OLD ONE!)](https://github.com/fchollet/keras/tree/1.2.2/keras) (*NOT THE MOST RECENT VERSION*)
    - set `image_dim_ordering : th` in `~/keras/keras.json`
    - set `backend : theano`, too.
  - [Kapre OLD VERSION for OLD KERAS](https://github.com/keunwoochoi/kapre/tree/a3bde3e38f62fc5458231198ea2528b752fbb373) In short,
  
```
$ pip install theano==0.9
$ pip install keras==1.2.2
$ git clone https://github.com/keunwoochoi/kapre.git
$ cd kapre
$ git checkout a3bde3e
$ python setup.py install
```

## Usage
```
$ python easy_feature_extraction.py audio_paths.txt some/path/features.npy
```
where `audio_path.txt` is line-by-line audio paths and `some/path/features.npy` is the path to save the result.

E.g., `audio_path.txt` : 
```
blah/a.mp3
blahblah/234.wav
some/other.c.mp3
```

Then load the `.npy` file. The features are size of `(num_songs, 160)`.


# Mode 2/2. To reproduce the paper
## Prerequisites

* Download datasets:
  - [Extended ballroom](http://anasynth.ircam.fr/home/media/ExtendedBallroom/)
  - [Gtzan genre](http://marsyasweb.appspot.com/download/data_sets/)
  - [Gtzan speech/music](http://marsyasweb.appspot.com/download/data_sets/)
  - [emoMusic](http://cvml.unige.ch/databases/emoMusic/)
  - [Jamendo singing voice](http://www.mathieuramona.com/wp/data/jamendo/)
  - [Urbansound8K](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html)

* Prerequisites
  - [Theano](http://deeplearning.net/software/theano/index.html). I used version 0.9.0 but should work with some similar versions.
  - [Keras 1.2.2 (OLD ONE!)](https://github.com/fchollet/keras/tree/1.2.2/keras) (*NOT THE MOST RECENT VERSION*)
    - set `image_dim_ordering : th` in `~/keras/keras.json`
    - set `backend : theano`, too.
  - [Kapre OLD VERSION for OLD KERAS](https://github.com/keunwoochoi/kapre/tree/a3bde3e38f62fc5458231198ea2528b752fbb373) by

```
$ git clone https://github.com/keunwoochoi/kapre.git
$ cd kapre
$ git checkout a3bde3e
$ python setup.py install
```
  - Optionally, `Sckikt learn, Pandas, Numpy`,.. for your convenience.

## Usage

* `0. main_prepare_many_datasets.ipynb`: prepare dataset, pre-processing
* `1. feature extraction for 6 tasks.ipynb`: feature extraction (MFCC and convnet features)
* `2_main_knn_svm_transfer`: Do SVM
* `3. knn and svm (with AveragePooling) results plots`: Plot results

# Appendix
## Links
 - [Train/valid/test split of MSD](https://github.com/keunwoochoi/MSD_split_for_tagging/blob/master/README.md) that I used for the training
 - [Paper: arXiv 1703.09179, Transfer Learning for Music Classification and Regression tasks](https://arxiv.org/abs/1703.09179)
 - [Blog article](https://keunwoochoi.wordpress.com/2017/03/28/paper-is-out-transfer-learning-for-music-classification-and-regression-tasks-and-behind-the-scene-negative-results-etc/) 

## Citation:
```
@inproceedings{choi2017transfer,
  title={Transfer learning for music classification and regression tasks},
  author={Choi, Keunwoo and Fazekas, George and Sandler, Mark and Cho, Kyunghyun},
  booktitle={The 18th International Society of Music Information Retrieval (ISMIR) Conference 2017, Suzhou, China},
  year={2017},
  organization={International Society of Music Information Retrieval}
}
```
  
