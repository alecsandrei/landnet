from __future__ import annotations

import argparse
import concurrent.futures
import functools
import io
import zipfile
from pathlib import Path

import requests

from landnet.config import RAW_DATA_DIR


def handle_link(link: str, out_dir: Path):
    print(link)
    out_dir_name = link.split('/')[-1].split('.')[0]
    out_archive = out_dir / out_dir_name
    if out_archive.exists():
        print(f'Skipping {out_archive} as it already exists.')
        return
    out_archive.mkdir()
    content = requests.get(link).content
    zip_content = zipfile.ZipFile(io.BytesIO(content))
    zip_content.extractall(out_archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download LAKI at target directory'
    )
    parser.add_argument(
        'path',
        type=Path,
        help='Path to the input file or directory.',
    )

    args = parser.parse_args()
    if not args.path.exists():
        raise NotADirectoryError(f'{args.path} is not a directory')
    laki = RAW_DATA_DIR / 'laki.txt'
    with laki.open(mode='r+') as file:
        links = list(map(str.strip, file.readlines()))
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        executor.map(functools.partial(handle_link, out_dir=args.path), links)
