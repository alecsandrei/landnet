from __future__ import annotations

import concurrent.futures
import io
import zipfile
from pathlib import Path

import requests

from landnet.config import RAW_DATA_DIR


def handle_link(link: str):
    print(link)
    content = requests.get(link).content
    out_dir = Path('/media/alex/My Passport/LiDAR/LAKI')
    out_dir_name = link.split('/')[-1].split('.')[0]
    zip_content = zipfile.ZipFile(io.BytesIO(content))
    out_archive = out_dir / out_dir_name
    out_archive.mkdir(exist_ok=True)
    zip_content.extractall(out_archive)


if __name__ == '__main__':
    laki = RAW_DATA_DIR / 'laki.txt'
    with laki.open(mode='r+') as file:
        links = list(map(str.strip, file.readlines()))
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        executor.map(handle_link, links)
