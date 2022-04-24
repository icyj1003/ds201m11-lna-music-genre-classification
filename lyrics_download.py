import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import pandas as pd


def nctGenres():
    return ['nhac-tre', 'tru-tinh',
            'rap-viet', 'tien-chien', 'rock-viet', 'cach-mang', 'thieu-nhi']


# Duong dan
audio_path = "./audio/"
data_path = "./data/"


def pick(url):

    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')

    return soup


def gather(url):

    try:
        soup = pick(url)

        pattern = re.compile(r"var strEcho = '(.*?)';$",
                             re.MULTILINE | re.DOTALL)

        script = soup.find("script", text=pattern)

        script = pattern.search(script.text).group(1)

        songs = BeautifulSoup(script, 'html.parser')

        return [song.get('href') for song in songs.find_all('a', class_='avatar')]

    except:
        return []


def extract(url):

    albums = pick(url).find_all('li', class_='list-album-item')
    albums = [album.find('a').get('href') for album in albums]

    return albums


def peel(url, genre):

    lyric = None

    title = None

    artis = None

    try:

        song = pick(url)

        lyric = song.find(
            'div', class_='content_lyrics dsc-body').text.strip().rstrip('Xem hết')

        title = song.find('h1', class_='name_detail').text.strip()

        if len(lyric.split('\n')) < 3:
            lyric = None

    except:

        pass

    return (title, url, lyric, genre)


def collect():

    url = input('URL: ')

    num = int(input('Pages: '))

    count = 0

    gList = nctGenres()

    for item in gList:

        print(item, count)

        count = count + 1

    genre = gList[int(input('Genre: '))]

    lyrics = []

    with tqdm(total=num) as pbar:

        for i in range(1, num + 1):

            albums = extract(url + f'?p={i}')

            for album in albums:

                songs = gather(album)

                for song in songs:

                    result = peel(song, genre)

                    if result[2] != None:

                        lyrics.append(result)

                df = pd.DataFrame(
                    lyrics, columns=['title', 'url', 'lyric', 'genre'])

                df.drop_duplicates()

                df.to_csv(data_path + f'{genre}.csv',
                          index=False, encoding='utf-8-sig')

            pbar.update(1)

    df = pd.DataFrame(
        lyrics, columns=['title', 'url', 'lyric', 'genre'])

    df.drop_duplicates()

    df.to_csv(data_path + f'{genre}.csv',
              index=False, encoding='utf-8-sig')


collect()
