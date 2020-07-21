import torch

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


class TFHider():
    tf = None
    def __init__(self):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow.python.util.deprecation as deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False

        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        TFHider.tf = tf


def classifier_loader():
    TFHider()

    gpus_list = TFHider.tf.config.experimental.list_physical_devices('GPU')
    TFHider.tf.config.experimental.set_visible_devices(gpus_list[torch.cuda.current_device()], 'GPU')

    with TFHider.tf.gfile.GFile('/data/~/tencent-ml-images/model.pb', 'rb') as f:
        graph_def = TFHider.tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with TFHider.tf.Graph().as_default() as graph:
        TFHider.tf.import_graph_def(graph_def)

    return graph


def classify(images, model, adversarial_attack):
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    with TFHider.tf.Session(graph=model) as sess:
        logits = sess.run('import/logits/output:0', feed_dict={'import/Placeholder:0': images})
    outputs = torch.from_numpy(logits).cuda()
    return outputs


# registry.add_model(
#     Model(
#         name = 'resnet101-tencent-ml-images',
#         arch = 'resnet101',
#         transform = StandardTransform(img_resize_size=256, img_crop_size=224),
#         normalization = StandardNormalization(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         classifier_loader = classifier_loader,
#         eval_batch_size = 256,
#         classify = classify,
#     )
# )
