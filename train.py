import time
import pandas as pd
import json
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import *
import warnings
warnings.filterwarnings("ignore")

# path
model_path = './model/'
cache_path = './cache/'
history_path = './history/'


def load_train_lyric_from_cache(augmentation=0):

    if augmentation == 0:
        X_train = np.load(cache_path + 'lyric_train.npy')
        y_train = np.load(cache_path + 'train_label.npy')
    else:
        X_train = np.load(cache_path + 'train_lyric_ovs.npy')
        y_train = np.load(cache_path + 'train_label_ovs.npy')

    X_val = np.load(cache_path + 'lyric_val.npy')
    y_val = np.load(cache_path + 'val_label.npy')

    return X_train, y_train, X_val, y_val


def load_train_audio_from_cache(augmentation=0):

    if augmentation == 0:
        X_train = np.load(cache_path + 'train_audio.npy')
        y_train = np.load(cache_path + 'train_label.npy')
    else:
        X_train = np.load(cache_path + 'train_audio_ovs.npy')
        y_train = np.load(cache_path + 'train_label_ovs.npy')

    X_val = np.load(cache_path + 'val_audio.npy')
    y_val = np.load(cache_path + 'val_label.npy')

    return X_train, y_train, X_val, y_val


# setting
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
width = 128
height = 128
channels = 3
num_classes = 6
epochs = 100
batch_size = 128
model_list = ['simpleCNN', 'resnet18', 'resnet34', 'lstm', 'bilstm']

is_all = input('Train all? y/n: ')

if is_all == 'n':
    # Choose model
    print('Choose:')
    for index, name in zip(range(0, 5), model_list):
        print(index, '-', name)

    model_index = int(input('number:'))
    model_name = model_list[model_index]
    aug = ''
    is_augmentation = int(input('no:0/yes:1 - '))
    if is_augmentation == 1:
        aug = '_ovs'

    # init
    model = None
    X_train = None
    y_train = None
    X_val = None
    y_val = None

    # build model
    if model_index in [3, 4]:  # lyric channel
        # get number of words
        with open(cache_path + 'w2i.json', 'r', encoding='utf8') as f:
            data = json.load(f)
            num_words = len(data)
        # Loading data
        X_train, y_train, X_val, y_val = load_train_lyric_from_cache(
            augmentation=is_augmentation)
        embedding_matrix = np.load(cache_path + 'embeding_matrix.npy')
        # create model
        if model_index == 4:
            model = build_BiLSTM(max_len=500, embedding_dim=300, embedding_matrix=embedding_matrix,
                                 num_classes=num_classes, num_words=num_words)
        elif model_index == 3:
            model = build_LSTM(max_len=500, embedding_dim=300, embedding_matrix=embedding_matrix,
                               num_classes=num_classes, num_words=num_words)
    elif model_index in [0, 1, 2]:
        # Loading data
        X_train, y_train, X_val, y_val = load_train_audio_from_cache(
            augmentation=is_augmentation)
        # create model
        if model_index == 0:
            model = simple_CNN(input_shape=(
                width, height, channels), num_classes=num_classes)
        elif model_index == 1:
            model = build_Resnet18(input_shape=(
                width, height, channels), num_classes=num_classes)
        elif model_index == 2:
            model = build_Resnet34(input_shape=(
                width, height, channels), num_classes=num_classes)
    # callbacks
    checkpoint = ModelCheckpoint(
        model_path + f'{model_name+aug}.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')
    callbacks_list = [checkpoint, early_stop]

    # compute class weights
    y_w = []
    for item in y_train:
        y_w.append(np.argmax(item))

    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(y_w), y_w)

    class_weights = {i: class_weights[i] for i in range(num_classes)}

    # training
    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics="accuracy")
    start = time.time()
    history = model.fit(X_train, y_train, batch_size=batch_size,
                        epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks_list, class_weight=class_weights)
    print(f'Complete in {time.time() - start}!')
    history = pd.DataFrame(history.history)
    print('Saving training history...')
    history.to_csv(history_path + f'{model_name+aug}.csv', index=False)
    print('Saving final model...')
    model.save(model_path + f'{model_name+aug}_final.h5')

else:
    is_augmentation = int(input('no:0/yes:1 - '))
    aug = ''
    if is_augmentation == 1:
        aug = '_ovs'
    for model_index in range(0, 5):
        model_name = model_list[model_index]

        # inity
        model = None
        X_train = None
        y_train = None
        X_val = None
        y_val = None

        # build model
        if model_index in [3, 4]:  # lyric channel
            # get number of words
            with open(cache_path + 'w2i.json', 'r', encoding='utf8') as f:
                data = json.load(f)
                num_words = len(data)
            # Loading data
            X_train, y_train, X_val, y_val = load_train_lyric_from_cache(
                augmentation=is_augmentation)
            embedding_matrix = np.load(cache_path + 'embeding_matrix.npy')
            # create model
            if model_index == 4:
                model = build_BiLSTM(max_len=500, embedding_dim=300, embedding_matrix=embedding_matrix,
                                     num_classes=num_classes, num_words=num_words)
            elif model_index == 3:
                model = build_LSTM(max_len=500, embedding_dim=300, embedding_matrix=embedding_matrix,
                                   num_classes=num_classes, num_words=num_words)
        elif model_index in [0, 1, 2]:
            # Loading data
            X_train, y_train, X_val, y_val = load_train_audio_from_cache(
                augmentation=is_augmentation)
            # create model
            if model_index == 0:
                model = simple_CNN(input_shape=(
                    width, height, channels), num_classes=num_classes)
            elif model_index == 1:
                model = build_Resnet18(input_shape=(
                    width, height, channels), num_classes=num_classes)
            elif model_index == 2:
                model = build_Resnet34(input_shape=(
                    width, height, channels), num_classes=num_classes)
        # callbacks
        checkpoint = ModelCheckpoint(
            model_path + f'{model_name+aug}.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stop = EarlyStopping(
            monitor='val_accuracy', patience=10, mode='max')
        callbacks_list = [checkpoint, early_stop]

        # compute class weights
        y_w = []
        for item in y_train:
            y_w.append(np.argmax(item))

        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(y_w), y_w)

        class_weights = {i: class_weights[i] for i in range(num_classes)}

        # training
        print(model.summary())

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics="accuracy")
        start = time.time()
        history = model.fit(X_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks_list, class_weight=class_weights)
        print(f'Complete in {time.time() - start}!')
        history = pd.DataFrame(history.history)
        print('Saving training history...')
        history.to_csv(history_path + f'{model_name+aug}.csv', index=False)
        print('Saving final model...')
        model.save(model_path + f'{model_name+aug}_final.h5')
