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

sys.path.append(str((pathlib.Path(__file__).parent / '../').resolve()))

import dataframe
import download_data


@click.command()
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def plot_grid(output_dir, skip_download):

    if skip_download:
        filename = join(output_dir, 'grid_df.pkl')
        if not exists(filename):
            raise Exception(f'Downloaded data not found at {filename}. Please run python src/plotting/download_data.py first')
        df = pd.read_pickle(filename)
    else:
        df = download_data.download_plotting_data(output_dir, store_data=True, verbose=True)


    df = dataframe.strip_metadata(df)

    sns.set(rc={'figure.figsize':(df.shape[1]//1.5, df.shape[0]//1.5)})
    plt.figure()
    heatmap = sns.heatmap(df, annot=True, cmap='viridis', cbar=False, square=True)
    heatmap.set_xlabel('Evaluation')
    heatmap.set_ylabel('Model')
    heatmap.set_title('ImageNet Accuracies')
    heatmap.figure.tight_layout()
    target = join(output_dir, f'full_grid.png')
    heatmap.figure.savefig(target)
    plt.close()
    print(f'Wrote the full grid to {target}')


    df = dataframe.replace_10percent_naive(df)

    sns.set(rc={'figure.figsize':(df.shape[1]//1.5, df.shape[0]//1.5)})
    plt.figure()
    heatmap = sns.heatmap(df, annot=True, cmap='viridis', cbar=False, square=True)
    heatmap.set_xlabel('Evaluation')
    heatmap.set_ylabel('Model')
    heatmap.set_title('ImageNet Accuracies')
    heatmap.figure.tight_layout()
    target = join(output_dir, f'collapsed_grid.png')
    heatmap.figure.savefig(target)
    plt.close()
    print(f'Wrote the collapsed grid to {target}')


    df = dataframe.aggregate_corruptions_naive(df)
    df = df[sorted(df.columns.to_list())]

    sns.set(rc={'figure.figsize':(df.shape[1]//1.2, df.shape[0]//1.5)})
    plt.figure()
    heatmap = sns.heatmap(df, annot=True, cmap='viridis', cbar=False, square=True)
    heatmap.set_xlabel('Evaluation')
    heatmap.set_ylabel('Model')
    heatmap.set_title('ImageNet Accuracies')
    heatmap.figure.tight_layout()
    target = join(output_dir, f'aggregated_grid.png')
    heatmap.figure.savefig(target)
    plt.close()
    print(f'Wrote the aggregated grid to {target}')


if __name__ == "__main__":
    plot_grid()