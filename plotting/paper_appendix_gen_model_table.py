import os
from os.path import join, exists
import argparse
import pathlib
from enum import Enum

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


def show_in_table(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name # and df_row.val >= 55


def use_for_line_fit(df_row):
    model_name, model_type, in_table = df_row.name.lower(), df_row.model_type, df_row.show_in_table
    return 'aws' not in model_name and 'batch64' not in model_name and model_type is NatModelTypes.STANDARD and in_table


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
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def generate_xy_plots(output_dir, output_file_dir, skip_download):

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

    df = prepare_df_for_plotting(df, df_metadata, ['val', 'imagenetv2-matched-frequency-format-val', 'avg_corruptions', 'avg_pgd'])
    df = plotter.add_plotting_data(df, ['val', 'imagenetv2-matched-frequency-format-val', 'avg_corruptions', 'avg_pgd'])

    df = format_eff_robust(df, 'val', 'imagenetv2-matched-frequency-format-val', 'logit')
    df = df.round(2)
    df = df[df.show_in_table]

    string = ""
    for i, (index, row) in enumerate(df.iterrows()):
        name = row.name
        name = name.replace('_', '\\_')
        string += f"\\foo{{{name}}} & {row.val} & {row['eff_robust']} & "
        string += f"{row.avg_corruptions if pd.notna(row.avg_corruptions) else ''} & {row.avg_pgd if pd.notna(row.avg_pgd) else ''} \\\\ \n"

    os.makedirs(output_file_dir, exist_ok=True)

    f = open(join(output_file_dir, f"model_table.tex"), "w+")
    f.write(string)
    f.close()
    print(f'written to {join(output_file_dir, f"model_table.tex")}')


def prepare_df_for_plotting(df, df_metadata, columns):
    assert set(columns).issubset(set(df.columns))

    df = df[columns]
    df_metadata = df_metadata[[x+'_dataset_size' for x in columns]]
    df = df.merge(df_metadata, right_index=True, left_index=True)
    # df = df.dropna()

    df['model_type'] = df.apply(get_model_type, axis=1)
    df['show_in_table'] = df.apply(show_in_table, axis=1)
    df['use_for_line_fit'] = df.apply(use_for_line_fit, axis=1)

    return df


if __name__ == '__main__':
    generate_xy_plots()
