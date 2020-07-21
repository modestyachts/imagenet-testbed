import os
from os.path import join, exists
import argparse
import pathlib

import click
import numpy as np
import pandas as pd
import scipy.stats

import download_data
import dataframe
import plotter
from plotter import transform_acc, inv_transform_acc
from model_types import NatModelTypes, nat_model_types_map


def get_model_type(df_row):
    return nat_model_types_map[df_row.name]


def show_in_plot1(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name and df_row.val >= 55

def show_in_plot2(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name and model_type != NatModelTypes.STANDARD # and df_row.val >= 55


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return 'aws' not in model_name and 'batch64' not in model_name and model_type is NatModelTypes.STANDARD and in_plot


def format_eff_robust(df, x_axis, y_axis, y_axis_fit, transform):
    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis_fit], transform)

    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    intercept, slope = lin_fit[1], lin_fit[0]
    lin_fit_ys_trans = transform_acc(df[x_axis], transform) * slope + intercept
    lin_fit_ys = inv_transform_acc(lin_fit_ys_trans, transform)

    df['eff_robust_y'] = df[y_axis_fit] - lin_fit_ys

    df_line = df[df.use_for_line_fit == True]
    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    intercept, slope = lin_fit[1], lin_fit[0]
    lin_fit_ys_trans = transform_acc(df[x_axis], transform) * slope + intercept
    lin_fit_ys = inv_transform_acc(lin_fit_ys_trans, transform)

    df['eff_robust_x'] = df[y_axis] - lin_fit_ys
    return df


@click.command()
@click.option('--x_axis', type=str, default='val')
@click.option('--y_axis', type=str, default='avg_corruptions')
@click.option('--y_axis_fit', type=str, default='imagenetv2-matched-frequency-format-val')
@click.option('--transform', type=str, default='logit')
@click.option('--num_bootstrap_samples', type=int, default=1000) #100000
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/figs').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def generate_xy_plot(x_axis, y_axis, y_axis_fit, transform, num_bootstrap_samples, output_dir, output_file_dir, skip_download):

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

    df = prepare_df_for_plotting(df, df_metadata, [x_axis, y_axis, y_axis_fit])
    df = plotter.add_plotting_data(df, [x_axis, y_axis, y_axis_fit])

    df = format_eff_robust(df, x_axis, y_axis, y_axis_fit, transform)

    # auto set xlim and ylim based on visible points
    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible[x_axis].min() - 1, df_visible[x_axis].max() + 0.5]
    ylim = [df_visible[y_axis].min() - 1, df_visible[y_axis].values.max() + 1]

    os.makedirs(output_file_dir, exist_ok=True)

    fig, _, legend = plotter.model_scatter_plot(df, x_axis, y_axis, xlim, ylim, NatModelTypes, 
                                         transform=transform, tick_multiplier=5, num_bootstrap_samples=num_bootstrap_samples,
                                         title='Distribution Shift to Corruptions Averaged', x_label='ImageNet', y_label='Corruptions Averaged', 
                                         figsize=(12, 8), include_legend=False, return_separate_legend=True)
    legend.savefig(join(output_file_dir, f'syn_shift_legend.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Legend saved to {join(output_file_dir, f'syn_shift_legend.pdf')}")
    fig.savefig(join(output_file_dir, f'syn_shift_corruptions.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_file_dir, f'syn_shift_corruptions.pdf')}")


    df.show_in_plot = df.apply(show_in_plot2, axis=1)

    # auto set xlim and ylim based on visible points
    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible['eff_robust_x'].min() - 1, df_visible['eff_robust_x'].max() + 1]
    ylim = [df_visible['eff_robust_y'].min() - 0.5, df_visible['eff_robust_y'].values.max() + 0.5]

    fig, _ = plotter.simple_scatter_plot(df, 'eff_robust_x', 'eff_robust_y', xlim, ylim, NatModelTypes, 
                                        title='Effective Robustness Scatterplot', 
                                        x_tick_multiplier=5, y_tick_multiplier=1,
                                        x_label='Corruptions Averaged Effective Robustness', y_label='ImageNetV2 Effective Robustness', 
                                        figsize=(12, 8), include_legend=False, return_separate_legend=False)

    fig.savefig(join(output_file_dir, f'eff_robust_corruptions.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_file_dir, f'eff_robust_corruptions.pdf')}")


def prepare_df_for_plotting(df, df_metadata, columns):
    assert set(columns).issubset(set(df.columns))

    df = df[columns]
    df_metadata = df_metadata[[x+'_dataset_size' for x in columns]]
    df = df.merge(df_metadata, right_index=True, left_index=True)
    df = df.dropna()

    df['model_type'] = df.apply(get_model_type, axis=1)
    df['show_in_plot'] = df.apply(show_in_plot1, axis=1)
    df['use_for_line_fit'] = df.apply(use_for_line_fit, axis=1)

    return df


if __name__ == '__main__':
    generate_xy_plot()
