import os
from os.path import join, exists
import pathlib
import subprocess
import sys

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import seaborn as sns


import dataframe
import download_data


@click.command()
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def plot_grid(output_dir, output_file_dir, skip_download):

    if skip_download:
        filename = join(output_dir, 'grid_df.pkl')
        if not exists(filename):
            raise Exception(f'Downloaded data not found at {filename}. Please run python src/plotting/download_data.py first')
        df = pd.read_pickle(filename)
    else:
        df = download_data.download_plotting_data(output_dir, store_data=True, verbose=True)

    print('NOTE: Please note this script will export raw top-1 model accuracies without any metadata. Use cautiously.')

    df = dataframe.strip_metadata(df)
    df = dataframe.replace_10percent_naive(df)

    df = df.drop(columns=['openimages_test_ilsvrc_subset', 'val-on-openimages-classes'])
    df = df[sorted(df.columns.to_list())]

    os.makedirs(output_file_dir, exist_ok=True)
    target = join(output_file_dir, f'robustness_top1s.csv')
    df.to_csv(target)
    print(f'Wrote the top-1 results csv to {target}')


if __name__ == "__main__":
    plot_grid()
