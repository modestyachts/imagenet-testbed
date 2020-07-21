import torch

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


model_params = {
'resnet50-vtab': {  'arch': 'resnet50',
                    'eval_batch_size': 128,
                    'img_crop_size': 224,
                    'img_resize_size': 256,
                    'output_node': 'logits'},
'resnet50-vtab-exemplar': { 'arch': 'resnet50',
                            'eval_batch_size': 128,
                            'img_crop_size': 224,
                            'img_resize_size': 256,
                            'output_node': 'classes'},
'resnet50-vtab-rotation': { 'arch': 'resnet50',
                            'eval_batch_size': 128,
                            'img_crop_size': 224,
                            'img_resize_size': 256,
                            'output_node': 'classes'},
'resnet50-vtab-semi-exemplar': {    'arch': 'resnet50',
                                    'eval_batch_size': 128,
                                    'img_crop_size': 224,
                                    'img_resize_size': 256,
                                    'img_resize_size': 256,
                                    'output_node': 'classes'},
'resnet50-vtab-semi-rotation': {    'arch': 'resnet50',
                                    'eval_batch_size': 128,
                                    'img_crop_size': 224,
                                    'img_resize_size': 256,
                                    'img_resize_size': 256,
                                    'output_node': 'classes'}}


class TFHider():
    tf = None
    def __init__(self):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow.python.util.deprecation as deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False

        import tensorflow as tf
        TFHider.tf = tf


def gen_classifier_loader(name, d):
    def classifier_loader():
        TFHider()
        gpus_list = TFHider.tf.config.experimental.list_physical_devices('GPU')
        TFHider.tf.config.experimental.set_visible_devices(gpus_list[torch.cuda.current_device()], 'GPU')
        loaded = TFHider.tf.saved_model.load('/data/~/vtab/' + name, tags=[])
        infer = loaded.signatures['default']
        return lambda images: infer(images)[d['output_node']]
    return classifier_loader


def classify(images, model, adversarial_attack):
    images = TFHider.tf.convert_to_tensor(images.cpu().numpy().transpose(0, 2, 3, 1))
    outputs = model(images)
    outputs = torch.from_numpy(outputs.numpy()).cuda()
    return outputs


# for name, d in model_params.items():
#     registry.add_model(
#         Model(
#             name = name,
#             arch = d['arch'],
#             transform = StandardTransform(d['img_resize_size'], d['img_crop_size']),
#             classifier_loader = gen_classifier_loader(name, d),
#             eval_batch_size = d['eval_batch_size'],
#             adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None,
#             classify = classify,
#         )
#     )
