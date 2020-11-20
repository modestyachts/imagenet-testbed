import timm
import torch
from torchvision import transforms
from PIL import Image

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


model_params = {
'vit_small_patch16_224': {  'arch': 'vit_small_patch16_224',
                            'eval_batch_size': 64,
                            'img_crop_size': 248,
                            'img_resize_size': 224,
                            'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225],
                            'qk_scale': 768 ** -0.5},
'vit_base_patch16_224': {   'arch': 'vit_base_patch16_224',
                            'eval_batch_size': 64,
                            'img_crop_size': 248,
                            'img_resize_size': 224,
                            'mean': [0.5, 0.5, 0.5],
                            'std': [0.5, 0.5, 0.5]},
'vit_base_patch16_384': {   'arch': 'vit_base_patch16_384',
                            'eval_batch_size': 64,
                            'img_crop_size': 384,
                            'img_resize_size': 384,
                            'mean': [0.5, 0.5, 0.5],
                            'std': [0.5, 0.5, 0.5]},
'vit_base_patch32_384': {   'arch': 'vit_base_patch32_384',
                            'eval_batch_size': 64,
                            'img_crop_size': 384,
                            'img_resize_size': 384,
                            'mean': [0.5, 0.5, 0.5],
                            'std': [0.5, 0.5, 0.5]},
'vit_large_patch16_224': {  'arch': 'vit_large_patch16_224',
                            'eval_batch_size': 64,
                            'img_crop_size': 248,
                            'img_resize_size': 224,
                            'mean': [0.5, 0.5, 0.5],
                            'std': [0.5, 0.5, 0.5]},                            
'vit_large_patch16_384': {  'arch': 'vit_large_patch16_384',
                            'eval_batch_size': 64,
                            'img_crop_size': 384,
                            'img_resize_size': 384,
                            'mean': [0.5, 0.5, 0.5],
                            'std': [0.5, 0.5, 0.5]},
'vit_large_patch32_384': {  'arch': 'vit_large_patch32_384',
                            'eval_batch_size': 64,
                            'img_crop_size': 384,
                            'img_resize_size': 384,
                            'mean': [0.5, 0.5, 0.5],
                            'std': [0.5, 0.5, 0.5]}}


def gen_classifier_loader(name, d):
    def classifier_loader():
        model = timm.create_model(name, pretrained=False, qk_scale=d['qk_scale'] if 'qk_scale' in d else None)
        load_model_state_dict(model, name)
        return model
    return classifier_loader


for name, d in model_params.items():
    registry.add_model(
        Model(
            name = name,
            arch = d['arch'],
            transform = transforms.Compose([transforms.Resize(d['img_crop_size'], interpolation=Image.BICUBIC),
                                            transforms.CenterCrop(d['img_resize_size']),
                                            transforms.ToTensor()]),
            normalization = StandardNormalization(d['mean'], d['std']),
            classifier_loader = gen_classifier_loader(name, d),
            eval_batch_size = d['eval_batch_size'],
            adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None
        )
    )
