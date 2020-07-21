import torchvision.models as torch_models

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


model_params = {
'resnet101_cutmix': {   'arch': 'resnet101',
                        'eval_batch_size': 256,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'input_space': 'RGB',
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'resnet50_cutmix': {   'arch': 'resnet50',
                       'eval_batch_size': 256,
                       'img_crop_size': 224,
                       'img_resize_size': 256,
                       'input_space': 'RGB',
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'resnet50_cutout': {   'arch': 'resnet50',
                       'eval_batch_size': 256,
                       'img_crop_size': 224,
                       'img_resize_size': 256,
                       'input_space': 'RGB',
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'resnet50_feature_cutmix': {   'arch': 'resnet50',
                               'eval_batch_size': 256,
                               'img_crop_size': 224,
                               'img_resize_size': 256,
                               'input_space': 'RGB',
                               'mean': [0.485, 0.456, 0.406],
                               'std': [0.229, 0.224, 0.225]},
'resnet50_mixup': {   'arch': 'resnet50',
                      'eval_batch_size': 256,
                      'img_crop_size': 224,
                      'img_resize_size': 256,
                      'input_space': 'RGB',
                      'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]}}


def gen_classifier_loader(name, d):
    def classifier_loader():
        model = torch_models.__dict__[d['arch']]()
        load_model_state_dict(model, name)
        return model
    return classifier_loader


for name, d in model_params.items():
    registry.add_model(
        Model(
            name = name,
            arch = d['arch'],
            transform = StandardTransform(d['img_resize_size'], d['img_crop_size']),
            normalization = StandardNormalization(d['mean'], d['std'], d['input_space']),
            classifier_loader = gen_classifier_loader(name, d),
            eval_batch_size = d['eval_batch_size'],
            adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None
        )
    )
