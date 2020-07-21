import torchvision.models as torch_models

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


model_params = {
'googlenet/inceptionv1': {   'arch': 'googlenet',
                             'eval_batch_size': 256,
                             'img_crop_size': 224,
                             'img_resize_size': 256,
                             'input_space': 'RGB',
                             'mean': [0.485, 0.456, 0.406],
                             'std': [0.229, 0.224, 0.225]},
'mnasnet0_5': {   'arch': 'mnasnet0_5',
                  'eval_batch_size': 256,
                  'img_crop_size': 224,
                  'img_resize_size': 256,
                  'input_space': 'RGB',
                  'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]},
'mnasnet1_0': {   'arch': 'mnasnet1_0',
                  'eval_batch_size': 256,
                  'img_crop_size': 224,
                  'img_resize_size': 256,
                  'input_space': 'RGB',
                  'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]},
'mobilenet_v2': {   'arch': 'mobilenet_v2',
                    'eval_batch_size': 256,
                    'img_crop_size': 224,
                    'img_resize_size': 256,
                    'input_space': 'RGB',
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]},
'resnext101_32x8d': {   'arch': 'resnext101_32x8d',
                        'eval_batch_size': 256,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'input_space': 'RGB',
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'resnext50_32x4d': {   'arch': 'resnext50_32x4d',
                       'eval_batch_size': 256,
                       'img_crop_size': 224,
                       'img_resize_size': 256,
                       'input_space': 'RGB',
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]},
'shufflenet_v2_x0_5': {   'arch': 'shufflenet_v2_x0_5',
                          'eval_batch_size': 256,
                          'img_crop_size': 224,
                          'img_resize_size': 256,
                          'input_space': 'RGB',
                          'mean': [0.485, 0.456, 0.406],
                          'std': [0.229, 0.224, 0.225]},
'shufflenet_v2_x1_0': {   'arch': 'shufflenet_v2_x1_0',
                          'eval_batch_size': 256,
                          'img_crop_size': 224,
                          'img_resize_size': 256,
                          'input_space': 'RGB',
                          'mean': [0.485, 0.456, 0.406],
                          'std': [0.229, 0.224, 0.225]},
'wide_resnet101_2': {   'arch': 'wide_resnet101_2',
                        'eval_batch_size': 256,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'input_space': 'RGB',
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'wide_resnet50_2': {   'arch': 'wide_resnet50_2',
                       'eval_batch_size': 256,
                       'img_crop_size': 224,
                       'img_resize_size': 256,
                       'input_space': 'RGB',
                       'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}}


def gen_classifier_loader(name, d):
    def classifier_loader():
        if name == 'googlenet/inceptionv1':
            model = torch_models.__dict__[d['arch']](pretrained=False, aux_logits=False, transform_input=True)
        else:
            model = torch_models.__dict__[d['arch']](pretrained=False)
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
