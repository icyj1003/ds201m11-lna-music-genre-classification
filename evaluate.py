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

    return test_audio, test_lyric, test_label


def save_result(model, input, output, file_name):
    y_pred = model.predict(input)
    y_hat = []
    y_true = []

    for item in y_pred:
        y_hat.append(np.argmax(item))

    for item in test_label:
        y_true.append(np.argmax(item))

    acc[file_name] = accuracy_score(y_true, y_hat)
    f1[file_name] = f1_score(y_true, y_hat, average='macro')
    rc[file_name] = recall_score(y_true, y_hat, average='macro')
    pre[file_name] = precision_score(y_true, y_hat, average='macro')
    cm = confusion_matrix(y_true, y_hat)
    ax = sb.heatmap(cm, annot=True, fmt=".0f", cmap='YlGnBu')
    plt.savefig(f'./cm/{file_name}.png', transparent=False)
    plt.clf()


test_audio, test_lyric, test_label = load_test_data_from_cache()

print('Loading model...')
custom_objects = {"WeightedAverage": WeightedAverage}

acc = {}
f1 = {}
rc = {}
pre = {}

# Saving ensemble
ensemble_models = [file_name for file_name in os.listdir(
    model_path) if file_name.find('ensemble') != -1 and file_name.find('final') == -1]

for item in ensemble_models:
    model_name = item.split('.')[0]
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model(model_path + item)
    save_result(model, [test_lyric, test_audio], test_label, model_name)

# saving lyric model
lyric_models = [file_name for file_name in os.listdir(
    model_path) if file_name.find('lstm') != -1 and file_name.find('ensemble') == -1 and file_name.find('final') == -1]
for item in lyric_models:
    model_name = item.split('.')[0]
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model(model_path + item)
    save_result(model, test_lyric, test_label, model_name)

# saving audio model
audio_models = [file_name for file_name in os.listdir(
    model_path) if file_name.find('lstm') == -1 and file_name.find('ensemble') == -1 and file_name.find('final') == -1]

for item in audio_models:
    model_name = item.split('.')[0]
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model(model_path + item)
    save_result(model, test_audio, test_label, model_name)

with open(cache_path + 'acc.json', 'w') as f:
    json.dump(acc, f)
with open(cache_path + 'f1.json', 'w') as f:
    json.dump(f1, f)
with open(cache_path + 'rc.json', 'w') as f:
    json.dump(rc, f)
with open(cache_path + 'pre.json', 'w') as f:
    json.dump(pre, f)
