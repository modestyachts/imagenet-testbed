import os
from enum import Enum
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


class HypModelTypes(Enum):
    STANDARD = ('Standard ResNet50', 'tab:blue', 200)
    ROBUST = ('ResNet50, hypothetical robustness intervention', 'tab:green', 200)


models = [k for k, v in model_types_map.items() if v == ModelTypes.STANDARD]
models = [m for m in models if 'subsample' not in m and 'batch64' not in m]


def get_model_type(df_row):
    if df_row.name in models:
        return HypModelTypes.STANDARD


def show_in_plot(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return model_name == 'resnet50_aws_baseline'


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return model_type is HypModelTypes.STANDARD and 'aws' not in model_name


@click.command()
@click.option('--x_axis', type=str, default='val')
@click.option('--y_axis', type=str, default='imagenetv2-matched-frequency-format-val')
@click.option('--transform', type=str, default='logit')
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()))
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

    hyp_robust_model = df.loc['resnet50_aws_baseline'].copy()
    arrow_params = [hyp_robust_model['val'], hyp_robust_model['imagenetv2-matched-frequency-format-val']]
    hyp_robust_model.model_type = HypModelTypes.ROBUST
    hyp_robust_model['val'] -= 8
    hyp_robust_model['imagenetv2-matched-frequency-format-val'] -= 3
    arrow_params += [hyp_robust_model['val'], hyp_robust_model['imagenetv2-matched-frequency-format-val']]
    hyp_robust_model.name = 'resnet50_aws_baseline_robust'
    hyp_robust_model.use_for_line_fit = False
    df = df.append(hyp_robust_model)

    df_visible = df[df.show_in_plot == True]
    xlim = [55.522000013427736, 85.72600555419922]
    ylim = [51.43000002441406, 76.83999633789062]

    fig, ax, = plotter.hyp_model_scatter_plot_quadrants(df, x_axis, y_axis, xlim, ylim, HypModelTypes, 
                                         transform=transform, tick_multiplier=5,
                                         title='Hypothetical Robustness Intervention', alpha=0.8,
                                         x_label='ImageNet', y_label='ImageNetV2', arrow_params=arrow_params,
                                         figsize=(12, 8), include_legend=True, return_separate_legend=False)

    l = ax.legend(loc='upper left',
                  ncol=1,
                  bbox_to_anchor=(0, 1),
                  fontsize=plotter.legend_fontsize,
                  scatterpoints=1,
                  columnspacing=0,
                  handlelength=1.5,
                  borderpad=0.2)
    for x in l.legendHandles:
        x._sizes = [100]

    os.makedirs(output_file_dir, exist_ok=True)

    fig.savefig(join(output_file_dir, f'small_resnet50.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_file_dir, f'small_resnet50.pdf')}")


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


