import json
import os
import numpy as np
from sklearn import ensemble
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Concatenate
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=(12, 10))
plt.rcParams.update({'font.size': 12})

# path
model_path = './model/'
cache_path = './cache/'
backup_path = './backup/'

# load model and data


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


def load_test_data_from_cache():

    test_audio = np.load(cache_path + 'test_audio.npy')
    test_lyric = np.load(cache_path + 'lyric_test.npy')
    test_label = np.load(cache_path + 'test_label.npy')
    test_id = np.load(cache_path + 'test_id.npy')

    return test_audio, test_lyric, test_label, test_id


test_audio, test_lyric, test_label, test_id = load_test_data_from_cache()
custom_objects = {"WeightedAverage": WeightedAverage}

with tf.keras.utils.custom_object_scope(custom_objects):
    model = load_model(model_path + 'ensemble_bilstm_resnet18_ovs.h5')

print(model.layers[-4].get_weights()[0])


# for i in range(2459):
#     li = np.array([test_lyric[i]])
#     ai = np.array([test_audio[i]])

#     y_pred = model.predict([li, ai])
#     true = np.argmax(test_label[i])
#     pred = np.argmax(y_pred)
#     if true == 0 and pred == 2:
#         print(test_id[i])
