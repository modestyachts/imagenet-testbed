import os
from os.path import join, exists
import argparse
import pathlib

import click
import numpy as np
import pandas as pd

import download_data
import dataframe
import plotter
from model_types import ModelTypes, model_types_map


def get_model_type(df_row):
    return model_types_map[df_row.name]


def show_in_plot(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name and model_name != 'resnet50_with_defocus_blur_aws'


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return 'aws' not in model_name and 'batch64' not in model_name and model_type is ModelTypes.STANDARD and in_plot


@click.command()
@click.option('--x_axis', type=str, default='val')
@click.option('--y_axis', type=str, default='imagenetv2-matched-frequency-format-val')
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def generate_xy_plot(x_axis, y_axis, output_dir, skip_download):

    if skip_download:
        filename = join(output_dir, 'grid_df.pkl')
        if not exists(filename):
            raise Exception(f'Downloaded data not found at {filename}. Please run python src/plotting/download_data.py first')
        df = pd.read_pickle(filename)
    else:
        df = download_data.download_plotting_data(output_dir, store_data=True, verbose=True)

    df, df_metadata = dataframe.extract_metadata(df)
    df, df_metadata = dataframe.replace_10percent_with_metadata(df, df_metadata)
    df, df_metadata = dataframe.aggregate_corruptions_with_metadata(df, df_metadata)


    df = prepare_df_for_plotting(df, df_metadata, [x_axis, y_axis])
    df = plotter.add_plotting_data(df, [x_axis, y_axis])

    plotter.label_fontsize = 18
    plotter.legend_fontsize = 15
    plotter.tick_fontsize = 15

    # auto set xlim and ylim based on visible points
    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible[x_axis].min() - 2, df_visible[x_axis].max() + 2]
    ylim = [df_visible[y_axis].min() - 2, df_visible[y_axis].max() + 2]

    fig, _, = plotter.model_scatter_plot(df, x_axis, y_axis, xlim, ylim, ModelTypes, 
                                         transform='logit', tick_multiplier=5, num_bootstrap_samples=1000,
                                         title='ImageNet', x_label=x_axis, y_label=y_axis, 
                                         figsize=(12, 8), include_legend=True, return_separate_legend=False)
    
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(join(output_dir, f'{x_axis}_vs_{y_axis}.pdf'), dpi='figure')
    print(f"Plot saved to {join(output_dir, f'{x_axis}_vs_{y_axis}.pdf')}")


def prepare_df_for_plotting(df, df_metadata, columns):
    assert set(columns).issubset(set(df.columns))

    df = df[columns]
    df_metadata = df_metadata[[x+'_dataset_size' for x in columns]]
    df = df.merge(df_metadata, right_index=True, left_index=True)
    df = df.dropna()

    df['model_type'] = df.apply(get_model_type, axis=1)
    df['show_in_plot'] = df.apply(show_in_plot, axis=1)
    df['use_for_line_fit'] = df.apply(use_for_line_fit, axis=1)

    return df


if __name__ == '__main__':
    generate_xy_plot()
