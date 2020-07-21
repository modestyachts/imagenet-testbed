import os
from os.path import join, exists
import argparse
import pathlib

import click
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

import download_data
import dataframe
import plotter
from plotter import transform_acc, inv_transform_acc
from model_types import NatModelTypes, nat_model_types_map


def get_model_type(df_row):
    return nat_model_types_map[df_row.name]


def show_in_plot(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name # and df_row.val >= 55


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return 'aws' not in model_name and 'batch64' not in model_name and model_type is NatModelTypes.STANDARD and in_plot


def format_eff_robust(df, x_axis, y_axis_fit, transform):
    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis_fit], transform)

    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    intercept, slope = lin_fit[1], lin_fit[0]
    lin_fit_ys_trans = transform_acc(df[x_axis], transform) * slope + intercept
    lin_fit_ys = inv_transform_acc(lin_fit_ys_trans, transform)

    df['eff_robust'] = df[y_axis_fit] - lin_fit_ys
    return df


@click.command()
@click.option('--x_axis', type=str, default='val')
@click.option('--y_axis_fit', type=str, default='imagenetv2-matched-frequency-format-val')
@click.option('--transform', type=str, default='logit')
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/figs').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def generate_xy_plot(x_axis, y_axis_fit, transform, output_dir, output_file_dir, skip_download):

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

    df = prepare_df_for_plotting(df, df_metadata, [x_axis, y_axis_fit])
    df = plotter.add_plotting_data(df, [x_axis, y_axis_fit])

    df = format_eff_robust(df, x_axis, y_axis_fit, transform)
    eff_robust = df.eff_robust

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.set_xticks([18, 34, 50, 101, 152])
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Effective Robustness (ImageNetV2)')

    resnet = eff_robust[['resnet' + str(x) for x in [18, 34, 50, 101, 152]]]
    se_resnet = eff_robust[['se_resnet' + str(x) for x in [50, 101, 152]]]
    ssl_resnet = eff_robust[['resnet' + str(x) + '_ssl' for x in [18, 50]]]
    swsl_resnet = eff_robust[['resnet' + str(x) + '_swsl' for x in [18, 50]]]
    resnext = eff_robust[['resnext' + str(x) + '_32x4d' for x in [50, 101]]]
    ssl_resnext = eff_robust[['resnext' + str(x) + '_32x4d_ssl' for x in [50, 101]]]
    swsl_resnext = eff_robust[['resnext' + str(x) + '_32x4d_swsl' for x in [50, 101]]]

    ax.plot([18, 34, 50, 101, 152], resnet.values, label='resnet', c='blue')
    ax.plot([50, 101, 152], se_resnet.values, label='se_resnet', c='green')
    ax.plot([18, 50], ssl_resnet.values, '--', label='resnet_ssl', c='blue')
    ax.plot([18, 50], swsl_resnet.values, '-.', label='resnet_swsl', c='blue')
    ax.plot([50, 101], resnext.values, label='resnext (32x4d)', c='red')
    ax.plot([50, 101], ssl_resnext.values, '--', label='resnext_ssl (32x4d)', c='red')
    ax.plot([50, 101], swsl_resnext.values, '-.', label='resnext_swsl (32x4d)', c='red')
    # ax.scatter(152, eff_robust['resnet152-imagenet11k'])
    ax.legend()

    os.makedirs(output_file_dir, exist_ok=True)

    fig.savefig(join(output_dir, f'eff_robust_inspect_plot.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_dir, f'eff_robust_inspect_plot.pdf')}")


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
