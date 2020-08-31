import os
from enum import Enum
from pathlib import Path
from os.path import join, exists
import argparse
import pathlib

import click
import numpy as np
import pandas as pd

import download_data
import dataframe
import plotter

from matplotlib import pyplot as plt
import seaborn as sns

import dataframe
import plotter

plotter.legend_fontsize = 20
alpha = 0.7

class ModelTypes(Enum):
    # Models plotted in order (so first enum value is plotted first)
    STANDARD = ('Standard training', (0.5, 0.5, 0.5, 0.5), 50)
    MORE_DATA = ('Trained with more data', 'tab:green', 100)
    NO_SUBSAMPLE = ('No subsampling', 'tab:blue', (300, 1.0))
    SUBSAMPLE_2 = ('Subsample 1/2', 'tab:purple', (300, 1.0))
    SUBSAMPLE_4 = ('Subsample 1/4', 'tab:brown', (300, 1.0))
    SUBSAMPLE_8 = ('Subsample 1/8', 'tab:olive', (300, 1.0))
    SUBSAMPLE_16 = ('Subsample 1/16', 'tab:cyan', (300, 1.0))
    SUBSAMPLE_32 = ('Subsample 1/32', 'tab:orange', (300, 1.0))


model_types_map = {
'resnet50_imagenet_100percent_batch64_original_images': ModelTypes.NO_SUBSAMPLE,
'resnet50_imagenet_subsample_1_of_16_batch64_original_images': ModelTypes.SUBSAMPLE_16,
'resnet50_imagenet_subsample_1_of_2_batch64_original_images': ModelTypes.SUBSAMPLE_2,
'resnet50_imagenet_subsample_1_of_32_batch64_original_images': ModelTypes.SUBSAMPLE_32,
'resnet50_imagenet_subsample_1_of_4_batch64_original_images': ModelTypes.SUBSAMPLE_4,
'resnet50_imagenet_subsample_1_of_8_batch64_original_images': ModelTypes.SUBSAMPLE_8,

'FixPNASNet': ModelTypes.STANDARD,
'FixResNeXt101_32x48d': ModelTypes.MORE_DATA,
'FixResNeXt101_32x48d_v2': ModelTypes.MORE_DATA,
'FixResNet50': ModelTypes.STANDARD,
'FixResNet50_no_adaptation': ModelTypes.STANDARD,
'FixResNet50_v2': ModelTypes.STANDARD,
'alexnet': ModelTypes.STANDARD,
'bninception': ModelTypes.STANDARD,
'bninception-imagenet21k': ModelTypes.MORE_DATA,
'cafferesnet101': ModelTypes.STANDARD,
'densenet121': ModelTypes.STANDARD,
'densenet161': ModelTypes.STANDARD,
'densenet169': ModelTypes.STANDARD,
'densenet201': ModelTypes.STANDARD,
'dpn107': ModelTypes.MORE_DATA,
'dpn131': ModelTypes.STANDARD,
'dpn68': ModelTypes.STANDARD,
'dpn68b': ModelTypes.MORE_DATA,
'dpn92': ModelTypes.MORE_DATA,
'dpn98': ModelTypes.STANDARD,
'efficientnet-b0': ModelTypes.STANDARD,
'efficientnet-b0-autoaug': ModelTypes.STANDARD,
'efficientnet-b1': ModelTypes.STANDARD,
'efficientnet-b1-autoaug': ModelTypes.STANDARD,
'efficientnet-b2': ModelTypes.STANDARD,
'efficientnet-b2-autoaug': ModelTypes.STANDARD,
'efficientnet-b3': ModelTypes.STANDARD,
'efficientnet-b3-autoaug': ModelTypes.STANDARD,
'efficientnet-b4': ModelTypes.STANDARD,
'efficientnet-b4-autoaug': ModelTypes.STANDARD,
'efficientnet-b5': ModelTypes.STANDARD,
'efficientnet-b5-autoaug': ModelTypes.STANDARD,
'efficientnet-b5-randaug': ModelTypes.STANDARD,
'efficientnet-b6-autoaug': ModelTypes.STANDARD,
'efficientnet-b7-autoaug': ModelTypes.STANDARD,
'efficientnet-b7-randaug': ModelTypes.STANDARD,
'efficientnet-l2-noisystudent': ModelTypes.MORE_DATA,
'fbresnet152': ModelTypes.STANDARD,
'google_resnet101_jft-300M': ModelTypes.MORE_DATA,
'googlenet/inceptionv1': ModelTypes.STANDARD,
'inceptionresnetv2': ModelTypes.STANDARD,
'inceptionv3': ModelTypes.STANDARD,
'inceptionv4': ModelTypes.STANDARD,
'instagram-resnext101_32x16d': ModelTypes.MORE_DATA,
'instagram-resnext101_32x32d': ModelTypes.MORE_DATA,
'instagram-resnext101_32x48d': ModelTypes.MORE_DATA,
'instagram-resnext101_32x8d': ModelTypes.MORE_DATA,
'mnasnet0_5': ModelTypes.STANDARD,
'mnasnet1_0': ModelTypes.STANDARD,
'mobilenet_v2': ModelTypes.STANDARD,
'nasnetalarge': ModelTypes.STANDARD,
'nasnetamobile': ModelTypes.STANDARD,
'pnasnet5large': ModelTypes.STANDARD,
'polynet': ModelTypes.STANDARD,
'resnet101': ModelTypes.STANDARD,
'resnet101-tencent-ml-images': ModelTypes.MORE_DATA,
'resnet152': ModelTypes.STANDARD,
'resnet152-imagenet11k': ModelTypes.MORE_DATA,
'resnet18': ModelTypes.STANDARD,
'resnet18_ssl': ModelTypes.MORE_DATA,
'resnet18_swsl': ModelTypes.MORE_DATA,
'resnet34': ModelTypes.STANDARD,
'resnet50': ModelTypes.STANDARD,
'resnet50-vtab': ModelTypes.STANDARD,
'resnet50-vtab-exemplar': ModelTypes.STANDARD,
'resnet50-vtab-rotation': ModelTypes.STANDARD,
'resnet50-vtab-semi-exemplar': ModelTypes.STANDARD,
'resnet50-vtab-semi-rotation': ModelTypes.STANDARD,
'resnet50_aws_baseline': ModelTypes.STANDARD,
'resnet50_ssl': ModelTypes.MORE_DATA,
'resnet50_swsl': ModelTypes.MORE_DATA,
'resnext101_32x16d_ssl': ModelTypes.MORE_DATA,
'resnext101_32x4d': ModelTypes.STANDARD,
'resnext101_32x4d_ssl': ModelTypes.MORE_DATA,
'resnext101_32x4d_swsl': ModelTypes.MORE_DATA,
'resnext101_32x8d': ModelTypes.STANDARD,
'resnext101_32x8d_ssl': ModelTypes.MORE_DATA,
'resnext101_32x8d_swsl': ModelTypes.MORE_DATA,
'resnext101_64x4d': ModelTypes.STANDARD,
'resnext50_32x4d': ModelTypes.STANDARD,
'resnext50_32x4d_ssl': ModelTypes.MORE_DATA,
'resnext50_32x4d_swsl': ModelTypes.MORE_DATA,
'se_resnet101': ModelTypes.STANDARD,
'se_resnet152': ModelTypes.STANDARD,
'se_resnet50': ModelTypes.STANDARD,
'se_resnext101_32x4d': ModelTypes.STANDARD,
'se_resnext50_32x4d': ModelTypes.STANDARD,
'senet154': ModelTypes.STANDARD,
'shufflenet_v2_x0_5': ModelTypes.STANDARD,
'shufflenet_v2_x1_0': ModelTypes.STANDARD,
'squeezenet1_0': ModelTypes.STANDARD,
'squeezenet1_1': ModelTypes.STANDARD,
'vgg11': ModelTypes.STANDARD,
'vgg11_bn': ModelTypes.STANDARD,
'vgg13': ModelTypes.STANDARD,
'vgg13_bn': ModelTypes.STANDARD,
'vgg16': ModelTypes.STANDARD,
'vgg16_bn': ModelTypes.STANDARD,
'vgg19': ModelTypes.STANDARD,
'vgg19_bn': ModelTypes.STANDARD,
'wide_resnet101_2': ModelTypes.STANDARD,
'wide_resnet50_2': ModelTypes.STANDARD,
'xception': ModelTypes.STANDARD,
}


def get_model_type(df_row):
    if df_row.name in model_types_map:
        return model_types_map[df_row.name]


def show_in_plot(df_row):
    return True


def use_for_line_fit(df_row):
    model_type, in_plot = df_row.model_type, df_row.show_in_plot
    return model_type is ModelTypes.STANDARD


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


@click.command()
@click.option('--x_axis', type=str, default='val')
@click.option('--y_axis', type=str, default='imagenetv2-matched-frequency-format-val')
@click.option('--transform', type=str, default='logit')
@click.option('--num_bootstrap_samples', type=int, default=1000) 
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()))
@click.option('--skip_download', is_flag=True, type=bool)
def generate_xy_plot(x_axis, y_axis, transform, num_bootstrap_samples, output_dir, output_file_dir, skip_download):

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

    df_visible = df[df.show_in_plot == True]
    xlim = [df_visible[x_axis].min() - 1, df_visible[x_axis].max() + 1]
    ylim = [df_visible[y_axis].min() - 2, df_visible[y_axis].values.max() + 5]

    fig, ax = plotter.model_scatter_plot(df, x_axis, y_axis, xlim, ylim, ModelTypes, 
                                         transform=transform, tick_multiplier=5, num_bootstrap_samples=num_bootstrap_samples,
                                         title='Robustness for Subsampling ImageNet', x_label='ImageNet (iid-subsampled)', y_label='ImageNetV2 (iid-\nsubsampled)', 
                                         figsize=(12, 8), include_legend=True, return_separate_legend=False)

    l = ax.legend(loc='upper left',
                  ncol=2,
                  bbox_to_anchor=(0, 1),
                  fontsize=plotter.legend_fontsize,
                  scatterpoints=1,
                  columnspacing=0,
                  handlelength=1.5,
                  borderpad=0.2)
    for x in l.legendHandles:
        x._sizes = [100]
        x.set_alpha(0.8)

    os.makedirs(output_file_dir, exist_ok=True)

    fig.savefig(join(output_file_dir, f'subsample_iid.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved to {join(output_file_dir, f'subsample_iid.pdf')}")

if __name__ == '__main__':
    generate_xy_plot()
