import os
from os.path import join, exists
import argparse
import pathlib
from enum import Enum

import click
import numpy as np
import pandas as pd

import download_data
import dataframe
import plotter
from model_types import NatModelTypes, nat_model_types_map


class BareModelTypes(Enum):
    STANDARD = ('Standard training', 'tab:blue', 100, 0.3, 'o')
    ROBUST_INTV = ('Robustness intervention', 'tab:brown', 100, 0.3, 'o')
    MORE_DATA = ('Trained with more data', 'tab:green', 100, 0.3, 'o')
    ROBUST_INTV_BIG = ('_nolegend_', 'tab:brown', 200, 1.0, 's')
    MORE_DATA_BIG = ('_nolegend_', 'tab:green', 200, 1.0, 's')

def get_model_type(df_row):
    if df_row.name in ['resnet152-imagenet11k', 'efficientnet-l2-noisystudent']:
        return BareModelTypes.MORE_DATA_BIG
    elif df_row.name == 'resnet50_with_motion_blur_aws':
        return BareModelTypes.ROBUST_INTV_BIG
    elif nat_model_types_map[df_row.name] == NatModelTypes.MORE_DATA:
        return BareModelTypes.MORE_DATA
    elif nat_model_types_map[df_row.name] == NatModelTypes.ROBUST_INTV:
        return BareModelTypes.ROBUST_INTV
    elif nat_model_types_map[df_row.name] == NatModelTypes.STANDARD:
        return BareModelTypes.STANDARD


def show_in_plot(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name and df_row.val >= 55


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name, df_row.model_type, df_row.show_in_plot
    return 'aws' not in model_name and 'batch64' not in model_name and nat_model_types_map[model_name] is NatModelTypes.STANDARD and in_plot


@click.command()
@click.option('--x_axis', type=str, default='val')
@click.option('--y_axis', type=str, default='imagenetv2-matched-frequency-format-val')
@click.option('--transform', type=str, default='logit')
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/figs').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def generate_xy_plot(x_axis, y_axis, transform, output_dir, output_file_dir, skip_download):

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
    df = df.dropna()

    # auto set xlim and ylim based on visible points
    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible[x_axis].min() - 0.5, df_visible[x_axis].max() + 0.5]
    ylim = [df_visible[y_axis].min() - 1, df_visible[[x_axis, y_axis]].values.max() + 0.5]

    fig, ax = plotter.model_scatter_plot_bare(df, x_axis, y_axis, xlim, ylim, BareModelTypes, 
                                         transform=transform, tick_multiplier=5,
                                         title='Simplified Distribution Shift Plot', x_label='ImageNet', y_label='ImageNetV2', 
                                         figsize=(12, 9), include_legend=True, return_separate_legend=False)

    l = ax.legend(loc='upper left',
                  ncol=1,
                  bbox_to_anchor=(0., 1),
                  fontsize=plotter.legend_fontsize,
                  scatterpoints=1,
                  columnspacing=0,
                  handlelength=1.5,
                  borderpad=0.2)

    for x in l.legendHandles:
        x._sizes = [100]
        x.set_alpha(0.8)

    os.makedirs(output_file_dir, exist_ok=True)

    fig.savefig(join(output_file_dir, f'imagenetv2_bare.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_file_dir, f'imagenetv2_bare.pdf')}")


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
