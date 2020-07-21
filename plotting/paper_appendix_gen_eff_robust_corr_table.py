import os
from os.path import join, exists
import argparse
import pathlib

import click
import pprint
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
    return 'subsample' not in model_name and model_type != NatModelTypes.STANDARD # and df_row.val >= 55


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return 'aws' not in model_name and 'batch64' not in model_name and 'subsample' not in model_name and model_type is NatModelTypes.STANDARD


def calc_eff_robust_corr(df, transform, x_axis, y_axis, x_axis_fit, y_axis_fit):
    df = df.copy()
    df_line = df[df.use_for_line_fit == True]

    if (df_line[y_axis_fit] == 0).any():
        pivot = df_line[df_line[y_axis_fit] == 0][x_axis_fit][0]

        df_line1 = df_line[df_line[x_axis_fit] < pivot]
        x_acc_line_trans = transform_acc(df_line1[x_axis_fit], transform)
        y_acc_line_trans = transform_acc(df_line1[y_axis_fit], transform)

        lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
        intercept, slope = lin_fit[1], lin_fit[0]
        lin_fit_ys_trans = transform_acc(df[x_axis_fit], transform) * slope + intercept
        lin_fit_ys = inv_transform_acc(lin_fit_ys_trans, transform)

        df['eff_robust_y'] = df[y_axis_fit] - lin_fit_ys

        df_line2 = df_line[df_line[x_axis_fit] > pivot]
        x_acc_line_trans = transform_acc(df_line2[x_axis_fit], transform)
        y_acc_line_trans = transform_acc(df_line2[y_axis_fit], transform)

        lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
        intercept, slope = lin_fit[1], lin_fit[0]
        lin_fit_ys_trans = transform_acc(df[x_axis_fit], transform) * slope + intercept
        lin_fit_ys = inv_transform_acc(lin_fit_ys_trans, transform)

        df.loc[df[x_axis_fit] > pivot, 'eff_robust_y'] = (df[y_axis_fit] - lin_fit_ys)[df[x_axis_fit] > pivot]

    else:
        x_acc_line_trans = transform_acc(df_line[x_axis_fit], transform)
        y_acc_line_trans = transform_acc(df_line[y_axis_fit], transform)

        lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
        intercept, slope = lin_fit[1], lin_fit[0]
        lin_fit_ys_trans = transform_acc(df[x_axis_fit], transform) * slope + intercept
        lin_fit_ys = inv_transform_acc(lin_fit_ys_trans, transform)

        df['eff_robust_y'] = df[y_axis_fit] - lin_fit_ys

    df2 = df[[x_axis, y_axis, 'use_for_line_fit']].dropna()
    df_line2 = df2[df2.use_for_line_fit == True]

    x_acc_line_trans = transform_acc(df_line2[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line2[y_axis], transform)

    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    intercept, slope = lin_fit[1], lin_fit[0]
    lin_fit_ys_trans = transform_acc(df2[x_axis], transform) * slope + intercept
    lin_fit_ys = inv_transform_acc(lin_fit_ys_trans, transform)

    df.loc[df2.index, 'eff_robust_x'] = df2[y_axis] - lin_fit_ys
    df = df[df.show_in_plot][['eff_robust_x', 'eff_robust_y']].dropna()
    return scipy.stats.pearsonr(df['eff_robust_x'], df['eff_robust_y'])[0]


@click.command()
@click.option('--transform', type=str, default='logit')
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def generate_scores(transform, output_dir, output_file_dir, skip_download):

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

    df['model_type'] = df.apply(get_model_type, axis=1)
    df['show_in_plot'] = df.apply(show_in_plot, axis=1)
    df['use_for_line_fit'] = df.apply(use_for_line_fit, axis=1)

    syn_shifts = [c for c in df.columns if any(x in c for x in ['disk', 'memory', 'pgd', 'corruptions', 'stylized', 'greyscale'])]
    nat_shifts = ['imagenetv2', 'objectnet', 'vid-robust_pm0', 'ytbb_pm0', 'vid-robust_pm10', 'ytbb_pm10', 'imagenet-a']

    results_df = pd.DataFrame(columns=nat_shifts, index=syn_shifts, dtype=np.float64)
    for syn_shift in syn_shifts:
        results_df.loc[syn_shift, 'imagenetv2'] = calc_eff_robust_corr(df, transform, 'val', syn_shift,
                                                    'val', 'imagenetv2-matched-frequency-format-val')
        results_df.loc[syn_shift, 'objectnet'] = calc_eff_robust_corr(df, transform, 'val', syn_shift,
                                                    'val-on-objectnet-classes', 'objectnet-1.0-beta')
        results_df.loc[syn_shift, 'vid-robust_pm0'] = calc_eff_robust_corr(df, transform, 'val', syn_shift,
                                                    'val-on-vid-robust-classes', 'imagenet-vid-robust_pm0')
        results_df.loc[syn_shift, 'ytbb_pm0'] = calc_eff_robust_corr(df, transform, 'val', syn_shift,
                                                    'val-on-ytbb-robust-classes', 'ytbb-robust_pm0')
        results_df.loc[syn_shift, 'vid-robust_pm10'] = calc_eff_robust_corr(df, transform, 'val', syn_shift,
                                                    'imagenet-vid-robust_pm0', 'imagenet-vid-robust_pm10')
        results_df.loc[syn_shift, 'ytbb_pm10'] = calc_eff_robust_corr(df, transform, 'val', syn_shift,
                                                    'ytbb-robust_pm0', 'ytbb-robust_pm10')
        results_df.loc[syn_shift, 'imagenet-a'] = calc_eff_robust_corr(df, transform, 'val', syn_shift,
                                                    'val-on-imagenet-a-classes', 'imagenet-a')

    results_df = results_df.round(2)
    results_df.sort_index(inplace=True)
    string = ""
    for i, (index, row) in enumerate(results_df.iterrows()):
        name = row.name
        name = name.replace('_', '\\_')
        string += f"{name} & {row['imagenetv2']} & {row['objectnet']} & {row['vid-robust_pm0']} & {row['ytbb_pm0']} & {row['vid-robust_pm10']} & {row['ytbb_pm10']} & {row['imagenet-a']} \\\\ \n"

    os.makedirs(output_file_dir, exist_ok=True)

    f = open(join(output_file_dir, f"eff_robust_corr_table.tex"), "w+")
    f.write(string)
    f.close()
    print(f'written to {join(output_file_dir, f"eff_robust_corr_table.tex")}')

    for nat_shift in nat_shifts:
        print('\n', nat_shift)
        print(results_df.nlargest(columns=[nat_shift], n=10)[nat_shift])


if __name__ == '__main__':
    generate_scores()
