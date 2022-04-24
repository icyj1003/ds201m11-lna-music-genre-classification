from tqdm import tqdm
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import threading


def download(link, filelocation):
    r = requests.get(link, stream=True)
    with open(filelocation, 'wb') as f:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)


def createNewDownloadThread(link, filelocation):
    download_thread = threading.Thread(
        target=download, args=(link, filelocation))
    download_thread.start()


def getDownloadLink(url):

    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser').find_all('script')[-1]

    special_code = ''

    soup = soup.text.split('\n')

    for s in soup:
        if s.strip().startswith('sources:'):
            special_code = s
            break

    try:
        a = re.findall('https.*?mp3', special_code)[0].replace("\/", "/")
        return a
    except:
        return None


# Duong dan
audio_path = "./audio/"
data_path = "./data/"


nhacvn = pd.read_csv(data_path + 'vlyrics.csv')

with tqdm(total=nhacvn.shape[0], desc='Downloading...') as pbar:
    for id, url in zip(nhacvn.ID, nhacvn.url):

        link = getDownloadLink(url)

        if link != None:
            createNewDownloadThread(
                link, audio_path + f'{id}.mp3')  # Đa luồng
        pbar.update(1)

print("Done!!!")
