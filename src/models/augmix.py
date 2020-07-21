import torchvision.models as torch_models

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


def classifier_loader():
    model = torch_models.resnet50()
    load_model_state_dict(model, 'resnet50_augmix')
    return model


registry.add_model(
    Model(
        name = 'resnet50_augmix',
        arch = 'resnet50',
        transform = StandardTransform(img_resize_size=256, img_crop_size=224),
        normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier_loader = classifier_loader,
        eval_batch_size = 256,
    )
)
