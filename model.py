from asyncio.proactor_events import constants
from statistics import mode
from classification_models.tfkeras import Classifiers
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Input, Embedding, Bidirectional, Dropout, LSTM, MaxPooling2D
from keras.initializers import Constant
from keras.models import Model, Sequential


# AUDIO


def build_Resnet18(input_shape, num_classes):

    ResNet18, preprocess_input = Classifiers.get('resnet18')
    resnet18 = ResNet18(input_shape,
                        weights='imagenet', include_top=False)
    for layer in resnet18.layers:
        layer.trainable = True

    for layer in resnet18.layers:
        layer.trainable = True

    flat = Flatten()(resnet18.layers[-1].output)

    fc1 = Dense(1024, activation='relu')(flat)

    output = Dense(num_classes, activation='softmax')(fc1)

    model = Model(inputs=resnet18.inputs, outputs=output)

    return model


def build_Resnet34(input_shape, num_classes):

    ResNet18, preprocess_input = Classifiers.get('resnet34')
    resnet18 = ResNet18(input_shape,
                        weights='imagenet', include_top=False)
    for layer in resnet18.layers:
        layer.trainable = True

    for layer in resnet18.layers:
        layer.trainable = True

    flat = Flatten()(resnet18.layers[-1].output)

    fc1 = Dense(1024, activation='relu')(flat)

    drop1 = Dropout(0.1)(fc1)

    output = Dense(num_classes, activation='softmax')(drop1)

    model = Model(inputs=resnet18.inputs, outputs=output)

    return model

# LYRICS


def build_BiLSTM(max_len, num_words, embedding_dim, embedding_matrix, num_classes):

    input = Input(shape=(max_len,))
    emb = Embedding(input_dim=num_words,
                    output_dim=embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    trainable=True,
                    input_length=max_len)(input)

    lstm1 = Bidirectional(LSTM(128, return_sequences=False))(emb)
    drop1 = Dropout(0.1)(lstm1)
    output = Dense(num_classes, activation="softmax")(drop1)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def build_LSTM(max_len, num_words, embedding_dim, embedding_matrix, num_classes):

    input = Input(shape=(max_len,))
    emb = Embedding(input_dim=num_words,
                    output_dim=embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    trainable=True,
                    input_length=max_len)(input)

    lstm1 = LSTM(128, return_sequences=False)(emb)
    drop1 = Dropout(0.1)(lstm1)
    output = Dense(num_classes, activation="softmax")(drop1)

    model = Model(inputs=input, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def simple_CNN(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# STACKING
