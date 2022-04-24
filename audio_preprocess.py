from tqdm import tqdm
import librosa
import librosa.display
import numpy as np
import warnings
from PIL import Image
import os
import multiprocessing
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

# Duong dan
spectr_path = './spectrogram/'
audio_path = './audio/'
mel_path = spectr_path + 'mel/'
har_path = spectr_path + 'har/'
per_path = spectr_path + 'per/'

if not os.path.exists(mel_path):
    os.makedirs(mel_path)

if not os.path.exists(har_path):
    os.makedirs(har_path)

if not os.path.exists(per_path):
    os.makedirs(per_path)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def save_spec(audio_fname):

    try:
        # Load audio file
        id = audio_fname.split('.')[0]
        y, sr = librosa.load(audio_path + audio_fname,
                             offset=1.0, duration=20, sr=44100)

        # Save mel spectr
        S = librosa.feature.melspectrogram(
            y, sr=sr, hop_length=512, n_mels=128)

        # Convert to log scale (dB)
        log_S = librosa.amplitude_to_db(S)

        img = scale_minmax(log_S, 0, 255).astype(np.uint8)
        img = Image.fromarray(img).resize((128, 128), Image.ANTIALIAS)
        img.save(mel_path + id + '.png')
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
        img.save(har_path + id + '.png')

        img = scale_minmax(log_S_p, 0, 255).astype(np.uint8)
        img = Image.fromarray(img).resize((128, 128), Image.ANTIALIAS)
        img.save(per_path + id + '.png')

    except Exception as e:
        return False
    return True


if __name__ == '__main__':

    audio_files = os.listdir(audio_path)
    nb_workers = 20
    pool = multiprocessing.Pool(nb_workers)
    it = pool.imap_unordered(save_spec, audio_files)

    ok_cnt = 0
    fail_cnt = 0

    for res in tqdm(it, total=len(audio_files)):
        if res:
            ok_cnt += 1
        else:
            fail_cnt += 1

    pool.close()

    print('Generating spectrogram finished! Generated {}/{} images successfully'.format(ok_cnt, ok_cnt + fail_cnt))
