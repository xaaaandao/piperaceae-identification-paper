#!/usr/bin/env python

import click
import os
import pandas as pd
import pathlib
import re

@click.command()
@click.option('-f', '--file', type=str, required=True)
@click.option('-i', '--input', type=str, required=True)
@click.option(
    '--taxon',
    '-t',
    type=click.Choice(['specific_epithet', 'genus']),
    required=True
)
def main(file, input, taxon):
    if not os.path.exists(input):
        raise NotADirectoryError(f'dir not found{input}')

    if not os.path.exists(file):
        raise FileNotFoundError(f'file not found{file}')

    data = open(file)
    list_data = [l.replace('\n', '') for l in data.readlines()]
    data.close()

    header = list_data[0]
    header = header.split(',')
    index_header = header.index(taxon)
    print(index_header)

    list_dir = [p for p in pathlib.Path(input).glob('*') if p.is_dir()]
    print(list_dir[0].parent)
    filename_label = os.path.join(list_dir[0].parent, 'label2.txt')
    with open(filename_label, 'w') as file_label:
        for d in list_dir:
            dir_jpeg = os.path.join(d, 'jpeg')

            files_dir = [f for f in pathlib.Path(dir_jpeg).glob('*') if f.is_file()]
            if len(files_dir) == 0:
                raise FileNotFoundError(f'files not found in {dir_jpeg}')

            filename = files_dir[0].stem
            filename = re.sub(r'[\W_][A-Za-z0-9]*', '', filename)

            result = [l for l in list_data if filename in l]
            info = result[0].split(',')
            string_filelabel = f'\"{info[index_header]}\";\"{d.stem}\"'
            print(string_filelabel)
            file_label.write(string_filelabel)
        file_label.close()

if __name__ == '__main__':
    main()