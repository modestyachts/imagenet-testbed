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

    df = df[[x for x in df.columns if 'memory' in x or 'greyscale' in x or 'disk' in x]]
    df = df.drop(columns=[x+'_in-memory' for x in ['elastic_transform', 'gaussian_blur', 'impulse_noise', 'shot_noise', 'snow', 'speckle_noise']])
    df = df.drop(columns=[x+'_on-disk' for x in ['elastic_transform', 'gaussian_blur', 'impulse_noise', 'shot_noise', 'snow', 'speckle_noise']])
    df = df[sorted(df.columns, key = lambda x: x.rsplit('_')[-1]+x)]
    
    df = df.loc[[x for x in df.index if 'FixResNeXt101_32x48d_v2' in x or 'aws' in x or 'efficientnet-b8-advprop-autoaug' in x]]
    rows = sorted(df.index)
    rows.append(rows.pop(rows.index('resnet50_with_gaussian_noise_contrast_motion_blur_jpeg_compression_aws')))
    rows.append(rows.pop(rows.index('FixResNeXt101_32x48d_v2')))
    rows.append(rows.pop(rows.index('efficientnet-b8-advprop-autoaug')))
    rows.insert(1, rows.pop(rows.index('resnet50_with_greyscale_aws')))
    df = df.loc[rows]

    os.makedirs(output_file_dir, exist_ok=True)

    sns.set(rc={'figure.figsize':(df.shape[1], df.shape[0])})
    plt.figure()
    heatmap = sns.heatmap(df, annot=True, cmap='viridis', cbar=False, square=True)
    heatmap.set_xlabel('Evaluation')
    heatmap.set_ylabel('Model')
    heatmap.set_title('ImageNet Accuracies (top-1, %)')
    heatmap.figure.tight_layout()
    aggregated_target = join(output_file_dir, f'zoom_grid.pdf')
    heatmap.figure.savefig(aggregated_target)
    plt.close()
    print(f'Wrote the aggregated grid to {aggregated_target}')


if __name__ == "__main__":
    plot_grid()