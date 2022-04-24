import pandas as pd

# paths
data_path = "./data/"
spectr_path = './spectrogram/'
audio_path = './audio/'
lyric_path = './lyric/'

# load metadata
vlyrics = pd.read_csv(data_path + 'clean.csv')
vlyrics.dropna(axis=0, inplace=True)

llist = vlyrics['lyric'].values
ids = vlyrics['ID'].values

# write lyric to txt
for lyric, id in zip(llist, ids):
    with open(lyric_path + f'{id}.txt', 'w', encoding='utf8') as file:
        file.write(str(lyric))
        file.close()
