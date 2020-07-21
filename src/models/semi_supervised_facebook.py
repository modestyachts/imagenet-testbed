import torch

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


model_params = {
'resnet18_ssl': {   'arch': 'resnet18',
                    'eval_batch_size': 256,
                    'img_crop_size': 224,
                    'img_resize_size': 256,
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]},
'resnet18_swsl': {      'arch': 'resnet18',
                        'eval_batch_size': 256,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'resnet50_ssl': {       'arch': 'resnet50',
                        'eval_batch_size': 256,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'resnet50_swsl': {      'arch': 'resnet50',
                        'eval_batch_size': 256,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'resnext50_32x4d_ssl': {        'arch': 'resnext50_32x4d',
                                'eval_batch_size': 256,
                                'img_crop_size': 224,
                                'img_resize_size': 256,
                                'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]},
'resnext50_32x4d_swsl': {       'arch': 'resnext50_32x4d',
                                'eval_batch_size': 256,
                                'img_crop_size': 224,
                                'img_resize_size': 256,
                                'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]},
'resnext101_32x4d_ssl': {      'arch': 'resnext101_32x4d',
                                'eval_batch_size': 32,
                                'img_crop_size': 224,
                                'img_resize_size': 256,
                                'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]},
'resnext101_32x4d_swsl': {      'arch': 'resnext101_32x4d',
                                'eval_batch_size': 32,
                                'img_crop_size': 224,
                                'img_resize_size': 256,
                                'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]},
'resnext101_32x8d_ssl': {      'arch': 'resnext101_32x8d',
                                'eval_batch_size': 16,
                                'img_crop_size': 224,
                                'img_resize_size': 256,
                                'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]},
'resnext101_32x8d_swsl': {      'arch': 'resnext101_32x8d',
                                'eval_batch_size': 16,
                                'img_crop_size': 224,
                                'img_resize_size': 256,
                                'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]},
'resnext101_32x16d_ssl': {      'arch': 'resnext101_32x16d',
                                'eval_batch_size': 16,
                                'img_crop_size': 224,
                                'img_resize_size': 256,
                                'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]},
# 'resnext101_32x16d_swsl': {     'arch': 'resnext101_32x16d',
#                                 'eval_batch_size': 16,
#                                 'img_crop_size': 224,
#                                 'img_resize_size': 256,
#                                 'mean': [0.485, 0.456, 0.406],
#                                 'std': [0.229, 0.224, 0.225]}
}


def gen_classifier_loader(name, d):
    def classifier_loader():
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', name)
        load_model_state_dict(model, name)
        return model
    return classifier_loader


for name, d in model_params.items():
    registry.add_model(
        Model(
            name = name,
            arch = d['arch'],
            transform = StandardTransform(d['img_resize_size'], d['img_crop_size']),
            normalization = StandardNormalization(d['mean'], d['std']),
            classifier_loader = gen_classifier_loader(name, d),
            eval_batch_size = d['eval_batch_size'],
            adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None
        )
    )
