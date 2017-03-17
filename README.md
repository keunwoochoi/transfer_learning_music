# transfer_learning_music

Repo for paper "Transfer learning for music classification and regression tasks"

## Prequisite

* Download datasets:
  - [Extended ballroom](http://anasynth.ircam.fr/home/media/ExtendedBallroom/)
  - [Gtzan genre](http://marsyasweb.appspot.com/download/data_sets/)
  - [Gtzan speech/music](http://marsyasweb.appspot.com/download/data_sets/)
  - [emoMusic](http://cvml.unige.ch/databases/emoMusic/)
  - [Jamendo singing voice](http://www.mathieuramona.com/wp/data/jamendo/)
  - [Urbansound8K](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html)

* Software
  - [Keras 1.2.2](https://github.com/fchollet/keras/tree/1.2.2/keras)
  - [Kapre 0.0.3](https://github.com/keunwoochoi/kapre), or get the right version with pip
  - Sckikt learn, pandas.

## Usage

* `0. main_prepare_many_datasets.ipynb`: prepare dataset, pre-processing
* `1. feature extraction for 6 tasks.ipynb`: feature extraction (MFCC and convnet features)
* `2_main_knn_svm_transfer`: Do SVM
* `3. knn and svm (with AveragePooling) results plots`: Plot results

## Link
  Paper link will added soon.
  