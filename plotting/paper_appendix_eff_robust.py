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
from model_types import ModelTypes, model_types_map, NatModelTypes, nat_model_types_map


cur_model_types, cur_model_types_map = None, None


def get_model_type(df_row):
    return cur_model_types_map[df_row.name]


def show_in_plot(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name and model_type != cur_model_types.STANDARD # and df_row.val >= 55


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return 'aws' not in model_name and 'batch64' not in model_name and 'subsample' not in model_name and model_type is cur_model_types.STANDARD


def format_eff_robust(df, x_axis, y_axis, x_axis_fit, y_axis_fit, transform):
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

    x_acc_line_trans = transform_acc(df_line[x_axis], transform)
    y_acc_line_trans = transform_acc(df_line[y_axis], transform)

    lin_fit = scipy.stats.linregress(x_acc_line_trans, y_acc_line_trans)
    intercept, slope = lin_fit[1], lin_fit[0]
    lin_fit_ys_trans = transform_acc(df[x_axis], transform) * slope + intercept
    lin_fit_ys = inv_transform_acc(lin_fit_ys_trans, transform)

    df['eff_robust_x'] = df[y_axis] - lin_fit_ys
    return df


def generate_xy_plot(x_axis, y_axis, x_axis_fit, y_axis_fit, transform, num_bootstrap_samples, output_dir, 
                     output_file_dir, skip_download, x_label, y_label):

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

    df = prepare_df_for_plotting(df, df_metadata, [x_axis, y_axis, x_axis_fit, y_axis_fit])
    df = plotter.add_plotting_data(df, [x_axis, y_axis, x_axis_fit, y_axis_fit])

    df = format_eff_robust(df, x_axis, y_axis, x_axis_fit, y_axis_fit, transform)

    # dfp = df[df.show_in_plot][['eff_robust_x', 'eff_robust_y']].dropna()
    # print("PEARSONR:", scipy.stats.pearsonr(dfp['eff_robust_x'], dfp['eff_robust_y'])[0])

    # auto set xlim and ylim based on visible points
    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible['eff_robust_x'].min() - 1, df_visible['eff_robust_x'].max() + 1]
    ylim = [df_visible['eff_robust_y'].min() - 0.5, df_visible['eff_robust_y'].values.max() + 0.5]

    fig, _, legend = plotter.simple_scatter_plot(df, 'eff_robust_x', 'eff_robust_y', xlim, ylim, cur_model_types, 
                                        title='Effective Robustness Scatterplot', 
                                        x_tick_multiplier=5, y_tick_multiplier=1,
                                        x_label=f'{x_label} Effective Robustness', y_label=f'{y_label}\nEffective Robustness', 
                                        figsize=(12, 8), include_legend=False, return_separate_legend=True)

    os.makedirs(output_file_dir, exist_ok=True)

    name = f'eff_robust_legend.pdf' if len(cur_model_types) == 3 else f'eff_robust_legend2.pdf'
    legend.savefig(join(output_file_dir, name), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Legend saved to {join(output_file_dir, name)}")

    fig_name = f'eff_robust_{y_axis.split("_")[1]}_{y_axis_fit.replace("1.0", "1")}.pdf'
    fig.savefig(join(output_file_dir, fig_name), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_file_dir, fig_name)}")


def prepare_df_for_plotting(df, df_metadata, columns):
    assert set(columns).issubset(set(df.columns))

    columns = list(set(columns))
    df = df[columns]
    df_metadata = df_metadata[[x+'_dataset_size' for x in columns]]
    df = df.merge(df_metadata, right_index=True, left_index=True)
    df = df.dropna()

    df['model_type'] = df.apply(get_model_type, axis=1)
    df['show_in_plot'] = df.apply(show_in_plot, axis=1)
    df['use_for_line_fit'] = df.apply(use_for_line_fit, axis=1)

    return df


if __name__ == '__main__':
    for y_axis in ['avg_pgd', 'avg_corruptions']:

        cur_model_types, cur_model_types_map = NatModelTypes, nat_model_types_map

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='val',
                         y_axis_fit='imagenetv2-matched-frequency-format-val',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='ImageNetV2',
                        )

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='val-on-objectnet-classes',
                         y_axis_fit='objectnet-1.0-beta',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='ObjectNet',
                        )

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='val-on-imagenet-a-classes',
                         y_axis_fit='imagenet-a',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='ImageNet-A',
                        )

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='val-on-vid-robust-classes',
                         y_axis_fit='imagenet-vid-robust_pm0',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='ImageNet-Vid-Robust (pm-0)',
                        )

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='val-on-ytbb-robust-classes',
                         y_axis_fit='ytbb-robust_pm0',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='YTBB-Robust (pm-0)',
                        )

        cur_model_types, cur_model_types_map = ModelTypes, model_types_map

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='imagenet-vid-robust_pm0',
                         y_axis_fit='imagenet-vid-robust_pm10',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='ImageNet-Vid-Robust (pm-10)',
                        )

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='ytbb-robust_pm0',
                         y_axis_fit='ytbb-robust_pm10',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='YTBB-Robust (pm-10)',
                        )

        cur_model_types, cur_model_types_map = NatModelTypes, nat_model_types_map

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='val-on-imagenet-r-classes',
                         y_axis_fit='imagenet-r',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='ImageNet-R',
                        )

        generate_xy_plot(x_axis='val',
                         y_axis=y_axis,
                         x_axis_fit='val',
                         y_axis_fit='imagenet-sketch',
                         transform='logit',
                         num_bootstrap_samples=1000, #100000
                         output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                         output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                         skip_download=True,
                         x_label='Lp Attacks' if 'pgd' in y_axis else 'Corruptions Averaged',
                         y_label='ImageNet-Sketch',
                        )
