import os
import random
import pandas as pd

# Duong dan
audio_path = "./audio/"
data_path = "./data/"
final_name = "vlyrics.csv"

parts = []
dir = os.listdir(path=data_path)

for file in dir:
    parts.append(file.replace('.csv', ''))

header = ['title', 'singer', 'lyric', 'genre']
df = pd.DataFrame()
count = 0
for file in dir:
    print(file)
    if file != final_name:
        part = pd.read_csv(data_path + f'{file}')
        part.drop_duplicates(subset=['lyric'], inplace=True)
        df = pd.concat([df, part], ignore_index=True)
        count = count + 1
        print(f"Đã hợp nhất {count} files!")
print(df.shape)

original_ids = range(0, df.shape[0])

while True:
    new_ids = {id_: random.randint(10_000_000, 99_999_999)
               for id_ in original_ids}
    if len(set(new_ids.values())) == len(original_ids):
        break

df['ID'] = pd.Series(original_ids).map(new_ids)

df.to_csv(data_path + final_name, index=False, encoding='utf-8-sig')
