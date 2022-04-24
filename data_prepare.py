from functools import cache
import matplotlib.pyplot as plt
import seaborn as sb
from turtle import update
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from json import load
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json


def i2l(index):
    mdict = {0: 'nhac-tre', 1: 'tru-tinh', 2: 'rap-viet',
             3: 'cach-mang', 4: 'thieu-nhi', 5: 'rock-viet'}
    return mdict[index]


def l2i(label):
    mdict = {'nhac-tre': 0, 'tru-tinh': 1, 'rap-viet': 2,
             'cach-mang': 3, 'thieu-nhi': 4, 'rock-viet': 5}
    return mdict[label]


def encoding(X):
    sentences = X
    X = []
    for s in sentences:
        sent = []
        for w in s.split():
            try:
                sent.append(w2i[w])
            except:
                sent.append(w2i["UNK"])
        X.append(sent)

    X = pad_sequences(maxlen=max_len, sequences=X,
                      padding="post", value=w2i["PAD"])
    return X


def decoding(vec):
    sent = ''
    for item in vec:
        sent = sent + i2w[item] + ' '
    return sent.strip()


def save_images(ids, fname):

    imgArray = np.zeros((len(ids), 128, 128, 3), 'uint8')
    index = 0
    with tqdm(total=len(ids), desc=f'Saving {fname}') as pbar:
        for id in ids:
            temp = np.zeros((128, 128, 3), 'uint8')
            mel_array = np.array(Image.open(mel_path + id + '.png'))
            temp[..., 0] = mel_array

            har_array = np.array(Image.open(har_path + id + '.png'))
            temp[..., 1] = har_array

            per_array = np.array(Image.open(per_path + id + '.png'))
            temp[..., 2] = per_array
            imgArray[index] = temp
            index += 1

            pbar.update(1)

    np.save(cache_path + fname, imgArray)


# Duong dan
data_path = "./data/"
spectr_path = './spectrogram/'
audio_path = './audio/'
lyric_path = './lyric/'
cache_path = './cache/'
mel_path = spectr_path + 'mel/'
har_path = spectr_path + 'har/'
per_path = spectr_path + 'per/'

# load metadata
print('Loading metadata...')
vlyrics = pd.read_csv(data_path + 'clean.csv')
ids = vlyrics['ID'].to_list()
labels = vlyrics['genre']
lyrics = vlyrics['lyric']

# load spectr fnames
mel_files = os.listdir(mel_path)
har_files = os.listdir(har_path)
per_files = os.listdir(per_path)

# setting
height = 128
width = 128
num_classes = 6
max_len = 500
embedding_dim = 300


# create dataset
X = []
y = []

with tqdm(total=len(ids), desc='Checking') as pbar:
    for id, label in zip(ids, labels):
        if str(id) + '.png' in mel_files and str(id) + '.png' in har_files and str(id) + '.png' in per_files:
            X.append(str(id))
            y.append(label)
        pbar.update(1)

# create output array
print(f'Creating output for {num_classes} classes...')
y = [l2i(item) for item in y]
y = to_categorical(y, num_classes=num_classes)

# train/test/val split
print('Spliting...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)


# Saving data
print('Saving id...')
np.save(cache_path + 'train_id.npy', X_train)
np.save(cache_path + 'test_id.npy', X_test)
np.save(cache_path + 'val_id.npy', X_val)


save_images(X_train, 'train_audio.npy')
save_images(X_test, 'test_audio.npy')
save_images(X_val, 'val_audio.npy')

print('Saving output ...')
np.save(cache_path + 'train_label.npy', np.array(y_train))
np.save(cache_path + 'test_label.npy', np.array(y_test))
np.save(cache_path + 'val_label.npy', np.array(y_val))


###############################################################
def load_lyrics(ids):
    temp_lyrics = []
    with tqdm(total=len(ids)) as pbar:
        for id in ids:
            with open(lyric_path + id + '.txt', 'r', encoding='utf8') as file:
                t = file.read()
                temp_lyrics.append(t)
            pbar.update(1)

    return temp_lyrics


print('Loading train lyrics...')
train_lyrics = load_lyrics(X_train)
print('Loading test lyrics...')
test_lyrics = load_lyrics(X_test)
print('Loading validation lyrics...')
val_lyrics = load_lyrics(X_val)

# Create vocabulary
print('Create vocabulary')
V = ''

V = ' '.join(train_lyrics)

V = list(set(V.split()))

w2i = {w: i + 2 for i, w in enumerate(V)}

w2i["UNK"] = 1
w2i["PAD"] = 0

num_words = len(V) + 2

i2w = {}

for k in w2i.keys():
    i2w[w2i[k]] = k

# save w2i i2w
with open(cache_path+'w2i.json', 'w', encoding='utf-8') as fp:
    json.dump(w2i, fp)

with open(cache_path + 'i2w.json', 'w', encoding='utf-8') as fp:
    json.dump(i2w, fp)

# encode and save lyric
train_lyrics_encoded = encoding(train_lyrics)
test_lyrics_encoded = encoding(test_lyrics)
val_lyrics_encoded = encoding(val_lyrics)

np.save(cache_path + 'lyric_train.npy', train_lyrics_encoded)
np.save(cache_path + 'lyric_test.npy', test_lyrics_encoded)
np.save(cache_path + 'lyric_val.npy', val_lyrics_encoded)


# Pretrain Word embedding
word_dict = []
embeddings_index = {}
f = open(data_path + 'word2vec_vi_words_300dims.txt', encoding='utf8')
print('Reading Pre-train Embedding...')
with tqdm(total=os.path.getsize(data_path + 'word2vec_vi_words_300dims.txt')) as pbar:

    for line in f:
        values = line.split(' ')
        word = values[0]
        word_dict.append(word)
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except Exception as e:
            pass
        pbar.update(len(line.encode('utf-8')))
    f.close()

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
with tqdm(total=len(w2i), desc='Creating Embedding Matrix') as pbar:
    for word, i in w2i.items():
        if i > 100000:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # doesn't exist, assign a random vector
            embedding_matrix[i] = np.random.randn(embedding_dim)
        pbar.update(1)

# Save embeding matrix
np.save(cache_path + 'embeding_matrix.npy', embedding_matrix)

print('All done!')
