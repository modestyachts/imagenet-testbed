import os
from os.path import join, exists
import argparse
import pathlib
from enum import Enum

import click
import numpy as np
import pandas as pd

from model_types import nat_model_types_map
import sys
sys.path.append('../src')
from models.cadene import model_params as cadene_params
from models.torchvision import model_params as torch_params


b = """\\begin{enumerate}
"""
e = """
\\end{enumerate}
"""

@click.command()
@click.option('--output_dir', type=str, default=str((pathlib.Path(__file__).parent / '../outputs').resolve()))
@click.option('--output_file_dir', type=str, default=str((pathlib.Path(__file__).parent / '../paper/appendix').resolve()))
def generate_xy_plots(output_dir, output_file_dir):
    string = b

    for name, model_type in nat_model_types_map.items():
        name2 = name.replace('_', '\\_')

        cite = ''
        url = ''
        if 'lpf' in name:
            cite = 'zhang2019making'
            url = 'https://github.com/adobe/antialiased-cnns'
        if 'SIN' in name:
            cite = 'geirhos2019imagenettrained'
            url = 'https://github.com/rgeirhos/texture-vs-shape'
        if 'smoothing_adversarial' in name:
            cite = 'salman2019provably'
            url = 'https://github.com/Hadisalman/smoothing-adversarial'
        if 'randomized_smoothing' in name:
            cite = 'cohen2019certified'
            url = 'https://github.com/locuslab/smoothing'
        if 'ssl' in name or 'swsl' in name:
            cite = 'yalniz2019billionscale'
            url = 'https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'
        if 'instagram' in name:
            cite = 'wslimageseccv2018'
            url = 'https://github.com/facebookresearch/WSL-Images'
        if 'resnet152-imagenet11k' in name:
            cite = 'imagenet11kmodel'
            url = 'https://github.com/tornadomeet/ResNet'
        if 'efficientnet' in name:
            cite = 'tan2019efficientnet'
            url = 'https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet'
        if 'autoaug' in name:
            cite = 'cubuk2018autoaugment'
            url = 'https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet'
        if 'randaug' in name:
            cite = 'cubuk2019randaugment'
            url = 'https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet'
        if 'advprop' in name:
            cite = 'xie2019adversarial'
            url = 'https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet'
        if 'noisystudent' in name:
            cite = 'xie2019selftraining'
            url = 'https://github.com/rwightman/pytorch-image-models'
        if 'Fix' in name:
            cite = 'touvron2019FixRes'
            url = 'https://github.com/facebookresearch/FixRes'
        if 'augmix' in name:
            cite = 'hendrycks2019augmix'
            url = 'https://github.com/google-research/augmix'
        if 'deepaugment' in name:
            cite = 'hendrycks2020faces'
            url = 'https://github.com/hendrycks/imagenet-r'
        if 'vtab' in name:
            cite = 'zhai2019visual'
            url = 'https://tfhub.dev/s?publisher=vtab'
        if 'resnet18-rotation' in name:
            cite = 'engstrom2019exploring'
            url = 'https://github.com/MadryLab/spatial-pytorch'
        if 'facebook_adv' in name:
            cite = 'xie2019feature'
            url = 'https://github.com/facebookresearch/ImageNet-Adversarial-Training'
        if 'robust' in name:
            cite = 'robustness'
            url = 'https://github.com/MadryLab/robustness'
        if name in cadene_params.keys():
            url = 'https://github.com/Cadene/pretrained-models.pytorch'
            if 'vgg' in name:
                cite = 'vgg'
            if 'resnet' in name:
                cite = 'resnet'
            if 'alexnet' in name:
                cite = 'alexnet'
            if 'inceptionv1' in name:
                cite = 'inceptionv1'
            if 'dpn' in name:
                cite = 'chen2017dual'
            if 'densenet' in name:
                cite = 'huang2016densely'
            if 'inceptionv3' == name:
                cite = 'szegedy2015rethinking'
            if 'inceptionv4' == name:
                cite = 'szegedy2016inceptionv4'
            if name in ['nasnetalarge', 'nasnetamobile']:
                cite = 'zoph2017learning'
            if name == 'pnasnet5large':
                cite = 'liu2017progressive'
            if name == 'polynet':
                cite = 'zhang2016polynet'
            if 'resnext' in name:
                cite = 'xie2016aggregated'
            if name[:2] == 'se':
                cite = 'hu2017squeezeandexcitation'
            if name == 'xception':
                cite = 'chollet2016xception'
        if name in torch_params.keys():
            url = 'https://github.com/pytorch/vision/tree/master/torchvision/models'
            if 'inceptionv1' in name:
                cite = 'inceptionv1'
            if 'mnasnet' in name:
                cite = 'tan2018mnasnet'
            if 'mobilenet_v2' == name:
                cite = 's2018mobilenetv2'
            if 'resnext' in name:
                cite = 'xie2016aggregated'
            if name[:2] == 'se':
                cite = 'hu2017squeezeandexcitation'
            if 'shufflenet' in name:
                cite = 'ma2018shufflenet'
            if 'wide_resnet' in name:
                cite = 'zagoruyko2016wide'
        if 'squeezenet' in name:
            cite = 'i2016squeezenet'
            url = 'https://github.com/Cadene/pretrained-models.pytorch'
        if 'cutmix' in name:
            cite = 'yun2019cutmix'
            url = 'https://github.com/clovaai/CutMix-PyTorch'
        if 'bninception' in name:
            cite = 'ioffe2015batch'
        if name == 'bninception-imagenet21k':
            url = 'https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-21k-inception.md'
        if 'jft' in name:
            cite = 'sun2017revisiting'
        if 'tencent' in name:
            cite = 'Wu_2019'
            url = 'https://github.com/Tencent/tencent-ml-images'
        if 'free' in name:
            cite = 'shafahi2019adversarial'
            url = 'https://github.com/mahyarnajibi/FreeAdversarialTraining'
        if 'cutout' in name:
            cite = 'devries2017improved'
            url = 'https://github.com/clovaai/CutMix-PyTorch'
        if 'mixup' in name:
            cite = 'zhang2017mixup'
            url = 'https://github.com/clovaai/CutMix-PyTorch'
        if 'BiT' in name:
            cite = 'alex2019big'
            url = 'https://github.com/google-research/big_transfer'

        if not ('subsample' in name or 'aws' in name or '100percent' in name):
            cite = f" \\citep{{{cite}}}"
            url = f"\\footnotesize{{\\url{{{url}}}}}"

        string += f"\\item \\model{{{name2}}}{cite}. {model_type.value[0]} model. {url} \n"

    string += e

    os.makedirs(output_file_dir, exist_ok=True)

    with open(join(output_file_dir, f"model_list.tex"), "w+") as f:
        f.write(string)
    print(f'List written to {join(output_file_dir, f"model_list.tex")}')


if __name__ == '__main__':
    generate_xy_plots()
