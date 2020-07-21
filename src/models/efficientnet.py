import PIL
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch
import numpy as np
import timm

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


model_params = {
'efficientnet-b0': {   'arch': 'efficientnet-b0',
                       'eval_batch_size': 200,
                       'img_size': 224,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b1': {   'arch': 'efficientnet-b1',
                       'eval_batch_size': 200,
                       'img_size': 240,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b2': {   'arch': 'efficientnet-b2',
                       'eval_batch_size': 200,
                       'img_size': 260,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b3': {   'arch': 'efficientnet-b3',
                       'eval_batch_size': 100,
                       'img_size': 300,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b4': {   'arch': 'efficientnet-b4',
                       'eval_batch_size': 100,
                       'img_size': 380,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b5': {   'arch': 'efficientnet-b5',
                       'eval_batch_size': 50,
                       'img_size': 456,
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'efficientnet-b0-autoaug': {   'arch': 'efficientnet-b0',
                               'eval_batch_size': 200,
                               'img_size': 224,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b1-autoaug': {   'arch': 'efficientnet-b1',
                               'eval_batch_size': 200,
                               'img_size': 240,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b2-autoaug': {   'arch': 'efficientnet-b2',
                               'eval_batch_size': 200,
                               'img_size': 260,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b3-autoaug': {   'arch': 'efficientnet-b3',
                               'eval_batch_size': 100,
                               'img_size': 300,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b4-autoaug': {   'arch': 'efficientnet-b4',
                               'eval_batch_size': 100,
                               'img_size': 380,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b5-autoaug': {   'arch': 'efficientnet-b5',
                               'eval_batch_size': 50,
                               'img_size': 456,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b6-autoaug': {   'arch': 'efficientnet-b6',
                               'eval_batch_size': 25,
                               'img_size': 528,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b7-autoaug': {   'arch': 'efficientnet-b7',
                               'eval_batch_size': 25,
                               'img_size': 600,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b5-randaug': {   'arch': 'efficientnet-b5',
                               'eval_batch_size': 50,
                               'img_size': 456,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b7-randaug': {   'arch': 'efficientnet-b7',
                               'eval_batch_size': 25,
                               'img_size': 600,
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'efficientnet-b0-advprop-autoaug': {    'arch': 'efficientnet-b0',
                                        'eval_batch_size': 200,
                                        'img_size': 224,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b1-advprop-autoaug': {    'arch': 'efficientnet-b1',
                                        'eval_batch_size': 200,
                                        'img_size': 240,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b2-advprop-autoaug': {    'arch': 'efficientnet-b2',
                                        'eval_batch_size': 200,
                                        'img_size': 260,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b3-advprop-autoaug': {    'arch': 'efficientnet-b3',
                                        'eval_batch_size': 100,
                                        'img_size': 300,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b4-advprop-autoaug': {    'arch': 'efficientnet-b4',
                                        'eval_batch_size': 100,
                                        'img_size': 380,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b5-advprop-autoaug': {    'arch': 'efficientnet-b5',
                                        'eval_batch_size': 50,
                                        'img_size': 456,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b6-advprop-autoaug': {    'arch': 'efficientnet-b6',
                                        'eval_batch_size': 25,
                                        'img_size': 528,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b7-advprop-autoaug': {    'arch': 'efficientnet-b7',
                                        'eval_batch_size': 25,
                                        'img_size': 600,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]},
'efficientnet-b8-advprop-autoaug': {    'arch': 'efficientnet-b8',
                                        'eval_batch_size': 25,
                                        'img_size': 672,
                                        'mean': [0.5, 0.5, 0.5],
                                        'std': [0.5, 0.5, 0.5]}}


CROP_PADDING = 32


def gen_classifier_loader(name, d):
    def classifier_loader():
        model = EfficientNet.from_name(d['arch'])
        load_model_state_dict(model, name)
        return model
    return classifier_loader

for name, d in model_params.items():
    registry.add_model(
        Model(
            name = name,
            arch = d['arch'],
            transform = transforms.Compose([transforms.Resize(d['img_size'] + CROP_PADDING, interpolation=PIL.Image.BICUBIC),
                                            transforms.CenterCrop(d['img_size']), 
                                            transforms.ToTensor()]),
            normalization = StandardNormalization(d['mean'], d['std']),
            classifier_loader = gen_classifier_loader(name, d),
            eval_batch_size = d['eval_batch_size'],
            adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None
        )
    )


def noisystudent_loader():
    model = timm.create_model('tf_efficientnet_l2_ns', pretrained=False)
    load_model_state_dict(model, 'efficientnet-l2-noisystudent')
    return model

registry.add_model(
    Model(
        name = 'efficientnet-l2-noisystudent',
        arch = 'efficientnet-l2',
        transform = transforms.Compose([transforms.Resize(800 + CROP_PADDING, interpolation=PIL.Image.BICUBIC),
                                        transforms.CenterCrop(800), 
                                        transforms.ToTensor()]),
        normalization = StandardNormalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        classifier_loader = noisystudent_loader,
        eval_batch_size = 1,
    )
)
