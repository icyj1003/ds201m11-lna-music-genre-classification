import time
from types import MemberDescriptorType
import pandas as pd
import json
from sklearn import ensemble
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import *
import warnings
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense
from keras.layers import concatenate, Concatenate, Layer
from tensorflow.keras.models import load_model


warnings.filterwarnings("ignore")


# path
model_path = './model/'
backup_path = './backup/'
cache_path = './cache/'
history_path = './history/'


def load_train_lyric_from_cache(augmentation=0):

    if augmentation == 1:
        X_train = np.load(cache_path + 'train_lyric_ovs.npy')
    else:
        X_train = np.load(cache_path + 'lyric_train.npy')
    X_val = np.load(cache_path + 'lyric_val.npy')

    return X_train, X_val


def load_train_audio_from_cache(augmentation=0):

    if augmentation == 1:
        X_train = np.load(cache_path + 'train_audio_ovs.npy')
    else:
        X_train = np.load(cache_path + 'train_audio.npy')
    X_val = np.load(cache_path + 'val_audio.npy')
    return X_train, X_val


def load_label_from_cache(augmentation=0):
    if augmentation == 1:
        y_train = np.load(cache_path + 'train_label_ovs.npy')
    else:
        y_train = np.load(cache_path + 'train_label.npy')
    y_val = np.load(cache_path + 'val_label.npy')
    return y_train, y_val


class WeightedAverage(Layer):

    def __init__(self, **kwargs):
        super(WeightedAverage, self).__init__()

    def build(self, input_shape):

        self.W = self.add_weight(
            shape=(1, 1, len(input_shape)),
            initializer='uniform',
            dtype=tf.float32,
            trainable=True)

    def call(self, inputs):

        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [tf.expand_dims(i, -1) for i in inputs]
        inputs = Concatenate(axis=-1)(inputs)  # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1)  # (1,1,n_inputs)
        # weights sum up to one on last dim

        return tf.reduce_sum(weights*inputs, axis=-1)  # (n_batch, n_feat)


def define_stacked_model(members, num_classes):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    # merge = concatenate(ensemble_outputs)
    merge = WeightedAverage()(ensemble_outputs)
    hidden = Dense(1024, activation='relu')(merge)
    drop = Dropout(0.5)(hidden)
    output = Dense(num_classes, activation='softmax')(drop)
    model = Model(inputs=ensemble_visible, outputs=output)
    # compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# setting

epochs = 100
batch_size = 128
num_classes = 6

# create
audio = ['resnet18', 'simpleCNN', 'resnet34']
lyric = ['bilstm', 'lstm']

for am in audio:
    for lm in lyric:
        for t in [1, 0]:
            aug = ""
            if t == 1:
                aug = '_ovs'
            members = [load_model(model_path + f'{lm+aug}.h5'),
                       load_model(model_path + f'{am+aug}.h5')]

            model = define_stacked_model(members=members, num_classes=6)
            train_audio, val_audio = load_train_audio_from_cache(
                augmentation=t)
            train_lyric, val_lyric = load_train_lyric_from_cache(
                augmentation=t)
            y_train, y_val = load_label_from_cache(augmentation=t)

            y_w = []
            for item in y_train:
                y_w.append(np.argmax(item))

            class_weights = class_weight.compute_class_weight(
                'balanced', np.unique(y_w), y_w)

            class_weights = {i: class_weights[i] for i in range(num_classes)}

            checkpoint = ModelCheckpoint(
                model_path + f'ensemble_{lm}_{am+aug}.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            early_stop = EarlyStopping(
                monitor='val_accuracy', patience=10, mode='max')
            callbacks_list = [checkpoint, early_stop]
            history = model.fit([train_lyric, train_audio], y_train,
                                validation_data=([val_lyric, val_audio], y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, class_weight=class_weights)
            history = pd.DataFrame(history.history)
            print('Saving training history...')
            history.to_csv(
                history_path + f'ensemble_{lm}_{am+aug}.csv', index=False)
            print('Saving final model...')
            model.save(model_path + f'ensemble_{lm}_{am+aug}_final.h5')
