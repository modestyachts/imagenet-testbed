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


class ModelTypes(Enum):
    STANDARD = ('Standard resnet50', 'tab:blue', 80)
    DATA_AUG = ('Trained with heavy data augmentation', 'tab:brown', 80)
    LP_ADV = ('Lp adversarially robust', 'tab:olive', 80)
    ARCH_MODIF = ('Architecture modification', 'tab:pink', 80)
    MORE_DATA = ('Trained with more data', 'tab:green', 80)

model_types_map = {
'FixResNet50': ModelTypes.ARCH_MODIF,
'FixResNet50CutMix': ModelTypes.DATA_AUG,
'FixResNet50CutMix_v2': ModelTypes.DATA_AUG,
'FixResNet50_no_adaptation': ModelTypes.ARCH_MODIF,
'FixResNet50_v2': ModelTypes.ARCH_MODIF,
'resnet50_aws_baseline': ModelTypes.STANDARD,
'resnet50-randomized_smoothing_noise_0.25': ModelTypes.LP_ADV,
'resnet50-randomized_smoothing_noise_0.50': ModelTypes.LP_ADV,
'resnet50-randomized_smoothing_noise_1.00': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_0.25': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_0.50': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_1.00': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_0.25': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_0.50': ModelTypes.LP_ADV,
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_1.00': ModelTypes.LP_ADV,
'resnet50-vtab-exemplar': ModelTypes.DATA_AUG,
'resnet50-vtab-rotation': ModelTypes.DATA_AUG,
'resnet50-vtab-semi-exemplar': ModelTypes.DATA_AUG,
'resnet50-vtab-semi-rotation': ModelTypes.DATA_AUG,
'resnet50_adv-train-free': ModelTypes.LP_ADV,
'resnet50_augmix': ModelTypes.DATA_AUG,
'resnet50_cutmix': ModelTypes.DATA_AUG,
'resnet50_cutout': ModelTypes.DATA_AUG,
'resnet50_feature_cutmix': ModelTypes.DATA_AUG,
'resnet50_l2_eps3_robust': ModelTypes.LP_ADV,
'resnet50_linf_eps4_robust': ModelTypes.LP_ADV,
'resnet50_linf_eps8_robust': ModelTypes.LP_ADV,
'resnet50_lpf2': ModelTypes.ARCH_MODIF,
'resnet50_lpf3': ModelTypes.ARCH_MODIF,
'resnet50_lpf5': ModelTypes.ARCH_MODIF,
'resnet50_mixup': ModelTypes.DATA_AUG,
'resnet50_ssl': ModelTypes.MORE_DATA,
'resnet50_swsl': ModelTypes.MORE_DATA,
'resnet50_trained_on_SIN': ModelTypes.DATA_AUG,
'resnet50_trained_on_SIN_and_IN': ModelTypes.DATA_AUG,
'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': ModelTypes.DATA_AUG,
'resnet50_with_brightness_aws': ModelTypes.DATA_AUG,
'resnet50_with_contrast_aws': ModelTypes.DATA_AUG,
# 'resnet50_with_defocus_blur_aws': ModelTypes.DATA_AUG,
'resnet50_with_fog_aws': ModelTypes.DATA_AUG,
'resnet50_with_frost_aws': ModelTypes.DATA_AUG,
'resnet50_with_gaussian_noise_aws': ModelTypes.DATA_AUG,
'resnet50_with_gaussian_noise_contrast_motion_blur_jpeg_compression_aws': ModelTypes.DATA_AUG,
'resnet50_with_greyscale_aws': ModelTypes.DATA_AUG,
'resnet50_with_jpeg_compression_aws': ModelTypes.DATA_AUG,
'resnet50_with_motion_blur_aws': ModelTypes.DATA_AUG,
'resnet50_with_pixelate_aws': ModelTypes.DATA_AUG,
'resnet50_with_saturate_aws': ModelTypes.DATA_AUG,
'resnet50_with_spatter_aws': ModelTypes.DATA_AUG,
'resnet50_with_zoom_blur_aws': ModelTypes.DATA_AUG,
'resnext50_32x4d': ModelTypes.ARCH_MODIF,
'resnext50_32x4d_ssl': ModelTypes.MORE_DATA,
'resnext50_32x4d_swsl': ModelTypes.MORE_DATA,
'se_resnet50': ModelTypes.ARCH_MODIF,
'se_resnext50_32x4d': ModelTypes.ARCH_MODIF,
'wide_resnet50_2': ModelTypes.ARCH_MODIF,

'FixPNASNet': ModelTypes.STANDARD,
'alexnet': ModelTypes.STANDARD,
'bninception': ModelTypes.STANDARD,
'cafferesnet101': ModelTypes.STANDARD,
'densenet121': ModelTypes.STANDARD,
'densenet161': ModelTypes.STANDARD,
'densenet169': ModelTypes.STANDARD,
'densenet201': ModelTypes.STANDARD,
'dpn131': ModelTypes.STANDARD,
'dpn68': ModelTypes.STANDARD,
'dpn98': ModelTypes.STANDARD,
'efficientnet-b0': ModelTypes.STANDARD,
'efficientnet-b1': ModelTypes.STANDARD,
'efficientnet-b2': ModelTypes.STANDARD,
'efficientnet-b3': ModelTypes.STANDARD,
'efficientnet-b4': ModelTypes.STANDARD,
'efficientnet-b5': ModelTypes.STANDARD,
'fbresnet152': ModelTypes.STANDARD,
'googlenet/inceptionv1': ModelTypes.STANDARD,
'inceptionresnetv2': ModelTypes.STANDARD,
'inceptionv3': ModelTypes.STANDARD,
'inceptionv4': ModelTypes.STANDARD,
'mnasnet0_5': ModelTypes.STANDARD,
'mnasnet1_0': ModelTypes.STANDARD,
'mobilenet_v2': ModelTypes.STANDARD,
'nasnetalarge': ModelTypes.STANDARD,
'nasnetamobile': ModelTypes.STANDARD,
'pnasnet5large': ModelTypes.STANDARD,
'polynet': ModelTypes.STANDARD,
'resnet101': ModelTypes.STANDARD,
'resnet152': ModelTypes.STANDARD,
'resnet18': ModelTypes.STANDARD,
'resnet34': ModelTypes.STANDARD,
'resnext101_32x4d': ModelTypes.STANDARD,
'resnext101_32x8d': ModelTypes.STANDARD,
'resnext101_64x4d': ModelTypes.STANDARD,
'se_resnet101': ModelTypes.STANDARD,
'se_resnet152': ModelTypes.STANDARD,
'se_resnext101_32x4d': ModelTypes.STANDARD,
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
'xception': ModelTypes.STANDARD,
}


def get_model_type(df_row):
    if df_row.name in model_types_map:
        return model_types_map[df_row.name]


def show_in_plot(df_row):
    model_name, model_type = df_row.name.lower(), df_row.model_type
    return ('resnet50' in model_name or 'resnext50' in model_name) # and df_row.val >= 57


def use_for_line_fit(df_row):
    model_name, model_type, in_plot = df_row.name.lower(), df_row.model_type, df_row.show_in_plot
    return model_type is ModelTypes.STANDARD 


def generate_xy_plot(x_axis, y_axis, transform, num_bootstrap_samples, output_dir, output_file_dir, skip_download, 
                     x_label, y_label, x_unit='top-1, %', y_unit='top-1, %', imagenet_a=False):

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
    xlim = [df_visible[x_axis].min() - 1, min(df_visible[x_axis].max() + 1, 99.5)]
    ylim = [df_visible[y_axis].min() - 1, df_visible[y_axis].values.max() + 1]

    os.makedirs(output_file_dir, exist_ok=True)

    if not imagenet_a:
        fig, _, legend = plotter.model_scatter_plot_quadrants(df, x_axis, y_axis, xlim, ylim, ModelTypes, 
                                         transform=transform, tick_multiplier=5, num_bootstrap_samples=num_bootstrap_samples,
                                         title='Relative and Effective Robustness - ResNet50 Family', alpha=0.8,
                                         x_label=x_label, y_label=y_label, x_unit=x_unit, y_unit=y_unit,
                                         figsize=(12, 8), include_legend=False, return_separate_legend=True)

        legend.savefig(join(output_file_dir, f'resnet50_legend.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
        print(f"Legend saved to {join(output_file_dir, f'resnet50_legend.pdf')}")
        fig.savefig(join(output_file_dir, f'resnet50_{y_axis}.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
        print(f"Plot saved to {join(output_file_dir, f'resnet50_{y_axis}.pdf')}")

    else:
        fig, _, legend = plotter.model_scatter_plot_quadrants_imagenet_a(df, x_axis, y_axis, xlim, ylim, ModelTypes, 
                                         transform=transform, tick_multiplier=5, num_bootstrap_samples=num_bootstrap_samples,
                                         title='Relative and Effective Robustness - ResNet50 Family', alpha=0.8,
                                         x_label=x_label, y_label=y_label, x_unit=x_unit, y_unit=y_unit, pivot=91.86,
                                         figsize=(12, 8), include_legend=False, return_separate_legend=True)

        legend.savefig(join(output_file_dir, f'resnet50_legend2.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
        print(f"Legend saved to {join(output_file_dir, f'resnet50_legend2.pdf')}")
        fig.savefig(join(output_file_dir, f'resnet50_{y_axis}.pdf'), dpi='figure', bbox_inches='tight', pad_inches=0.1)
        print(f"Plot saved to {join(output_file_dir, f'resnet50_{y_axis}.pdf')}")


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
    generate_xy_plot(x_axis='val',
                     y_axis='imagenetv2-matched-frequency-format-val',
                     transform='logit',
                     num_bootstrap_samples=1000, #100000
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet',
                     y_label='ImageNetV2',
                    )

    generate_xy_plot(x_axis='val-on-objectnet-classes',
                     y_axis='objectnet-1.0-beta',
                     transform='logit',
                     num_bootstrap_samples=1000, #100000
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet (class-subsampled)',
                     y_label='ObjectNet',
                    )

    generate_xy_plot(x_axis='val-on-vid-robust-classes',
                     y_axis='imagenet-vid-robust_pm0',
                     transform='logit',
                     num_bootstrap_samples=1000, #100000
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet (class-subsampled)',
                     y_label='ImageNet-Vid-Robust',
                     y_unit='pm-0, %',
                    )

    generate_xy_plot(x_axis='val-on-ytbb-robust-classes',
                     y_axis='ytbb-robust_pm0',
                     transform='logit',
                     num_bootstrap_samples=1000, #100000
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet (class-subsampled)',
                     y_label='YTBB-Robust',
                     y_unit='pm-0, %',
                    )

    generate_xy_plot(x_axis='imagenet-vid-robust_pm0',
                     y_axis='imagenet-vid-robust_pm10',
                     transform='logit',
                     num_bootstrap_samples=1000, #100000
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='ImageNet-Vid-Robust',
                     y_label='ImageNet-Vid-Robust',
                     x_unit='pm-0, %',
                     y_unit='pm-10, %',
                    )

    generate_xy_plot(x_axis='ytbb-robust_pm0',
                     y_axis='ytbb-robust_pm10',
                     transform='logit',
                     num_bootstrap_samples=1000, #100000
                     output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                     output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                     skip_download=True,
                     x_label='YTBB-Robust',
                     y_label='YTBB-Robust',
                     x_unit='pm-0, %',
                     y_unit='pm-10, %',
                    )

    generate_xy_plot(x_axis='val-on-imagenet-a-classes',
                 y_axis='imagenet-a',
                 transform='logit',
                 num_bootstrap_samples=1000, #100000
                 output_dir=str((pathlib.Path(__file__).parent / '../outputs').resolve()),
                 output_file_dir=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()),
                 skip_download=True,
                 x_label='ImageNet (class-subsampled)',
                 y_label='ImageNet-A',
                 imagenet_a=True,
                )
