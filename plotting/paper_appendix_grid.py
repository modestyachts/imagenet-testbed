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
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def plot_grid(output_dir, output_file_dir, skip_download):

    if skip_download:
        filename = join(output_dir, 'grid_df.pkl')
        if not exists(filename):
            raise Exception(f'Downloaded data not found at {filename}. Please run python src/plotting/download_data.py first')
        df = pd.read_pickle(filename)
    else:
        df = download_data.download_plotting_data(output_dir, store_data=True, verbose=True)

    df = dataframe.strip_metadata(df)
    df = dataframe.replace_10percent_naive(df)
    df = dataframe.aggregate_corruptions_naive(df)

    df = df.drop(columns=['val_subsampled_class_1_8', 'imagenetv2-matched-frequency-format-val_subsampled_class_1_8',
                          'openimages_test_ilsvrc_subset', 'val-on-openimages-classes'])
    df = df.drop([x for x in df.index if 'subsample' in x and 'classes' in x])

    df = df[sorted(df.columns.to_list())]

    os.makedirs(output_file_dir, exist_ok=True)

    sns.set(rc={'figure.figsize':(df.shape[1]//1.5, df.shape[0]//1.5 - 25)})
    plt.figure()
    heatmap = sns.heatmap(df, annot=True, cmap='viridis', cbar=False, square=True)
    heatmap.set_xlabel('Evaluation Setting')
    heatmap.set_ylabel('Model')
    heatmap.set_title('ImageNet Accuracies (top-1, %)')
    heatmap.figure.tight_layout()
    aggregated_target = join(output_file_dir, f'aggregated_grid.pdf')
    heatmap.figure.savefig(aggregated_target, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f'Wrote the aggregated grid to {aggregated_target}')


if __name__ == "__main__":
    plot_grid()