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


def show_in_plot1(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name and df_row.val >= 55

def show_in_plot2(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return 'subsample' not in model_name and model_type != cur_model_types.STANDARD # and df_row.val >= 55


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return 'aws' not in model_name and 'batch64' not in model_name and model_type is cur_model_types.STANDARD and in_plot


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


def generate_xy_plot(x_axis, y_axis, x_axis_fit, y_axis_fit, transform, num_bootstrap_samples, output_dir, output_file_dir, 
                    skip_download, x_label, y_label, y_label2):

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

    df = prepare_df_for_plotting(df, df_metadata, list(set([x_axis, y_axis, x_axis_fit, y_axis_fit])))
    df = plotter.add_plotting_data(df, list(set([x_axis, y_axis, x_axis_fit, y_axis_fit])))

    df = format_eff_robust(df, x_axis, y_axis, x_axis_fit, y_axis_fit, transform)

    # auto set xlim and ylim based on visible points
    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible[x_axis].min() - 1, df_visible[x_axis].max() + 0.5]
    ylim = [df_visible[y_axis].min() - 1, df_visible[y_axis].values.max() + 1]

    fig, _, legend = plotter.model_scatter_plot(df, x_axis, y_axis, xlim, ylim, cur_model_types, 
                                         transform=transform, tick_multiplier=5, num_bootstrap_samples=num_bootstrap_samples,
                                         title=f'Distribution Shift to {y_label}', x_label=x_label, y_label=y_label, 
                                         figsize=(12, 8), include_legend=False, return_separate_legend=True)

    os.makedirs(output_file_dir, exist_ok=True)

    filename = f'topk_plot_{y_axis.replace(".", "_")}_{y_axis_fit.replace("1.0", "1")}.pdf'
    fig.savefig(join(output_file_dir, filename), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_file_dir, filename)}")

    df.show_in_plot = df.apply(show_in_plot2, axis=1)

    # auto set xlim and ylim based on visible points
    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible['eff_robust_x'].min() - 1, df_visible['eff_robust_x'].max() + 1]
    ylim = [df_visible['eff_robust_y'].min() - 0.5, df_visible['eff_robust_y'].values.max() + 0.5]

    fig, _ = plotter.simple_scatter_plot(df, 'eff_robust_x', 'eff_robust_y', xlim, ylim, cur_model_types, 
                                        title='Effective Robustness Scatterplot', 
                                        x_tick_multiplier=5, y_tick_multiplier=1,
                                        x_label=f'{y_label} Effective Robustness', y_label=f'{y_label2}\nEffective Robustness', 
                                        figsize=(12, 8), include_legend=False, return_separate_legend=False)

    filename = f'topk_corr_{y_axis.replace(".", "_")}_{y_axis_fit.replace("1.0", "1")}.pdf'
    fig.savefig(join(output_file_dir, filename), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_file_dir, filename)}")


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
    cur_model_types, cur_model_types_map = NatModelTypes, nat_model_types_map

    generate_xy_plot(x_axis='val',
                     y_axis='brightness_on-disk',
                     x_axis_fit='val',
                     y_axis_fit='imagenetv2-matched-frequency-format-val',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Brightness (on-disk)',
                     y_label2 = 'ImagetNetV2'
                    )

    generate_xy_plot(x_axis='val',
                     y_axis='saturate_on-disk',
                     x_axis_fit='val',
                     y_axis_fit='imagenetv2-matched-frequency-format-val',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Saturate (on-disk)',
                     y_label2 = 'ImagetNetV2'
                    )


    generate_xy_plot(x_axis='val',
                     y_axis='contrast_on-disk',
                     x_axis_fit='val-on-objectnet-classes',
                     y_axis_fit='objectnet-1.0-beta',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Contrast (on-disk)',
                     y_label2 = 'ObjectNet'
                    )

    generate_xy_plot(x_axis='val',
                     y_axis='fog_on-disk',
                     x_axis_fit='val-on-objectnet-classes',
                     y_axis_fit='objectnet-1.0-beta',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Fog (on-disk)',
                     y_label2 = 'ObjectNet'
                    )


    generate_xy_plot(x_axis='val',
                     y_axis='gaussian_blur_on-disk',
                     x_axis_fit='val-on-vid-robust-classes',
                     y_axis_fit='imagenet-vid-robust_pm0',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Gaussian Blur (on-disk)',
                     y_label2='ImageNet-Vid-Robust (pm-0)',
                    )

    generate_xy_plot(x_axis='val',
                     y_axis='defocus_blur_on-disk',
                     x_axis_fit='val-on-vid-robust-classes',
                     y_axis_fit='imagenet-vid-robust_pm0',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Defocus Blur (on-disk)',
                     y_label2='ImageNet-Vid-Robust (pm-0)',
                    )


    generate_xy_plot(x_axis='val',
                     y_axis='defocus_blur_on-disk',
                     x_axis_fit='val-on-ytbb-robust-classes',
                     y_axis_fit='ytbb-robust_pm0',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Defocus Blur (on-disk)',
                     y_label2='YTBB-Robust (pm-0)',
                    )

    generate_xy_plot(x_axis='val',
                     y_axis='defocus_blur_in-memory',
                     x_axis_fit='val-on-ytbb-robust-classes',
                     y_axis_fit='ytbb-robust_pm0',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Defocus Blur (in-memory)',
                     y_label2='YTBB-Robust (pm-0)',
                    )


    generate_xy_plot(x_axis='val',
                     y_axis='contrast_on-disk',
                     x_axis_fit='val-on-imagenet-a-classes',
                     y_axis_fit='imagenet-a',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Contrast (on-disk)',
                     y_label2='ImageNet-A',
                    )

    generate_xy_plot(x_axis='val',
                     y_axis='contrast_in-memory',
                     x_axis_fit='val-on-imagenet-a-classes',
                     y_axis_fit='imagenet-a',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Contrast (in-memory)',
                     y_label2='ImageNet-A',
                    )


    cur_model_types, cur_model_types_map = ModelTypes, model_types_map

    generate_xy_plot(x_axis='val',
                     y_axis='pgd.linf.eps0.5',
                     x_axis_fit='imagenet-vid-robust_pm0',
                     y_axis_fit='imagenet-vid-robust_pm10',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='PGD (Linf Eps0.5)',
                     y_label2='ImageNet-Vid-Robust (pm-10)',
                    )

    generate_xy_plot(x_axis='val',
                     y_axis='spatter_in-memory',
                     x_axis_fit='imagenet-vid-robust_pm0',
                     y_axis_fit='imagenet-vid-robust_pm10',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Spatter (in-memory)',
                     y_label2='ImageNet-Vid-Robust (pm-10)',
                    )


    generate_xy_plot(x_axis='val',
                     y_axis='pgd.l2.eps0.5',
                     x_axis_fit='ytbb-robust_pm0',
                     y_axis_fit='ytbb-robust_pm10',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='PGD (L2 Eps0.5)',
                     y_label2='YTBB-Robust (pm-10)',
                    )

    generate_xy_plot(x_axis='val',
                     y_axis='speckle_noise_on-disk',
                     x_axis_fit='ytbb-robust_pm0',
                     y_axis_fit='ytbb-robust_pm10',
                     transform='logit',
                     num_bootstrap_samples=1000, 
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='Speckle Noise (on-disk))',
                     y_label2='YTBB-Robust (pm-10)',
                    )
