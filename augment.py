from collections import Counter
import os
import json
from time import time
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import warnings
from PIL import Image
import pickle
import warnings
warnings.filterwarnings("ignore")

# path
cache_path = './cache/'
spectr_path = './spectrogram/'
audio_path = './audio/'
mel_path = spectr_path + 'mel/'
har_path = spectr_path + 'har/'
per_path = spectr_path + 'per/'


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def oversamping_spec(id, times, overlap=5):
    audio_fname = id + '.mp3'
    imgArray = np.zeros((times, 128, 128, 3), 'uint8')
    for t in range(times):
        try:
            temp = np.zeros((128, 128, 3), 'uint8')

            # Load audio file
            id = audio_fname.split('.')[0]
            y, sr = librosa.load(audio_path + audio_fname,
                                 offset=1.0 + (t+1)*overlap, duration=20, sr=44100)

            # Save mel spectr
            S = librosa.feature.melspectrogram(
                y, sr=sr, hop_length=512, n_mels=128)

            # Convert to log scale (dB)
            log_S = librosa.amplitude_to_db(S)

            img = scale_minmax(log_S, 0, 255).astype(np.uint8)
            img = Image.fromarray(img).resize((128, 128), Image.ANTIALIAS)
            temp[..., 0] = np.array(img)

            # Save harmonic spectr and percussive spectr
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            S_h = librosa.feature.melspectrogram(
                y_harmonic, sr=sr, n_mels=128, hop_length=512)
            S_p = librosa.feature.melspectrogram(
                y_percussive, sr=sr, n_mels=128, hop_length=512)

            log_S_h = librosa.power_to_db(S_h, ref=np.max)
            log_S_p = librosa.power_to_db(S_p, ref=np.max)

            img = scale_minmax(log_S_h, 0, 255).astype(np.uint8)
            img = Image.fromarray(img).resize((128, 128), Image.ANTIALIAS)
            temp[..., 1] = np.array(img)

            img = scale_minmax(log_S_p, 0, 255).astype(np.uint8)
            img = Image.fromarray(img).resize((128, 128), Image.ANTIALIAS)
            temp[..., 2] = np.array(img)
            imgArray[t] = temp
        except:
            return None
    return imgArray


def modify_sentence(sentence, synonyms, p=0.5):
    for i in range(len(sentence)):
        if np.random.random() > p:
            try:
                syns = synonyms[sentence[i]]
                sentence[i] = np.random.choice(syns)
            except KeyError:
                pass
    return sentence


def oversampling_lyric(lyric, times):

    X_gen = np.array([modify_sentence(lyric, synonyms)
                     for _ in range(times)])
    return X_gen


# init
w2i = {}
i2w = {}

# Load data
with open(cache_path + 'w2i.json', 'r', encoding='utf8') as f:
    data = json.load(f)
    num_words = len(data)
    w2i = dict(data)

with open(cache_path + 'i2w.json', 'r', encoding='utf8') as f:
    data = json.load(f)
    i2w = dict(data)

synonyms_number = 5
word_number = num_words
embedding_matrix = np.load(cache_path + 'embeding_matrix.npy')

# Create synonyms
if not os.path.exists(cache_path+'synonyms.pkl'):

    nn = NearestNeighbors(n_neighbors=synonyms_number+1).fit(embedding_matrix)
    neighbours_mat = nn.kneighbors(embedding_matrix[1:word_number])[1]
    synonyms = {x[0]: x[1:] for x in neighbours_mat}
    # save synonyms
    with open(cache_path+'synonyms.pkl', 'wb') as fp:
        pickle.dump(synonyms, fp)

else:
    with open(cache_path+'synonyms.pkl', 'rb') as fp:
        synonyms = pickle.load(fp)


train_ids = np.load(cache_path + 'train_id.npy')
train_lyric = np.load(cache_path + 'lyric_train.npy')
train_label = np.load(cache_path + 'train_label.npy')
train_audio = np.load(cache_path + 'train_audio.npy')

new_train_lyric = train_lyric
new_train_label = train_label
new_train_audio = train_audio

class_count = Counter([np.argmax(i) for i in train_label])

max_num = max(list(class_count.values()))

class_rates = {}
for k, v in class_count.items():
    temp = None
    if v < max_num/4:
        temp = int(max_num/4/v)
    else:
        temp = 1
    class_rates[k] = temp

with tqdm(total=len(train_ids)) as pbar:
    for id, lyric, label in zip(train_ids, train_lyric, train_label):
        cr = class_rates[np.argmax(label)]
        if cr != 1:
            new_train_lyric = np.concatenate(
                [new_train_lyric, oversampling_lyric(lyric, cr)])
            new_train_label = np.concatenate(
                [new_train_label, np.array([label for _ in range(cr)])])
            new_train_audio = np.concatenate(
                [new_train_audio, oversamping_spec(id, cr)])
        pbar.update(1)

np.save(cache_path + 'train_audio_ovs', new_train_audio)
np.save(cache_path + 'train_lyric_ovs', new_train_lyric)
np.save(cache_path + 'train_label_ovs', new_train_label)

print(class_count)
print(Counter([np.argmax(i) for i in new_train_label]))
