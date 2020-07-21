import io
from inspect import signature
import torch
from torch import nn
from torchvision import transforms


class ConvertToPyTorchModel(nn.Module):
    def __init__(self, base_model, classify_fn_args, classify=None, normalization=None, 
                class_sublist=None, adversarial_attack=None):
        super().__init__()

        if normalization is not None:
            self.input_space = normalization.input_space
            self.mean = nn.Parameter(torch.tensor(normalization.mean).float().view(3, 1, 1))
            self.std = nn.Parameter(torch.tensor(normalization.std).float().view(3, 1, 1))

        self.base_model = base_model
        self.classify_fn_args = classify_fn_args
        self.classify = classify
        self.class_sublist = class_sublist
        self.adversarial_attack = adversarial_attack
        self.normalization = normalization

    def forward(self, x):
        if self.normalization is not None:
            if self.input_space == 'BGR':
                x = x.flip(1)
            x = (x - self.mean) / self.std

        if self.classify is None:
            x = self.base_model(x)

        else:
            kwargs = {'images': x, 'model': self.base_model}
            if 'class_sublist' in self.classify_fn_args:
                kwargs['class_sublist'] = self.class_sublist
            if 'adversarial_attack' in self.classify_fn_args:
                kwargs['adversarial_attack'] = self.adversarial_attack

            x = self.classify(**kwargs)

        if self.class_sublist is not None and 'class_sublist' not in self.classify_fn_args:
            x = x.t()[self.class_sublist].t()

        return x


class Model():
    ADVERSARIAL_BATCH_SIZE_REDUCTION_FACTOR = 8

    def __init__(self, name, transform, classifier_loader, eval_batch_size, arch=None,
                 normalization=None, classify=None, adversarial_batch_size=None):
        super().__init__()

        self.name = name
        self.arch = arch if arch is not None else 'NA'
        self.transform = transform
        self.classifier_loader = classifier_loader
        self.eval_batch_size = eval_batch_size
        self.adversarial_batch_size = adversarial_batch_size
        self.normalization = normalization
        self.classify = classify
        self.classify_fn_args = set()

        if self.classify is not None:
            sig = list(signature(self.classify).parameters.keys())
            assert 'images' in sig and 'model' in sig, 'Unrecognized metrics function ' + \
                'definition. Make sure function takes arguments "images" and "model"'

            for arg in ['images', 'model', 'class_sublist', 'adversarial_attack']:
                if arg in sig:
                    self.classify_fn_args.add(arg)

    def generate_classifier(self, py_eval_setting):
        self.classifier = self.classifier_loader()
        model = ConvertToPyTorchModel(self.classifier, self.classify_fn_args, self.classify, self.normalization,
                                      py_eval_setting.class_sublist, py_eval_setting.adversarial_attack)

        if len(list(model.parameters())) == 0:
            # stop pytorch from freaking out
            model._dummy = nn.Parameter(torch.tensor(0.0))

        return model

    def get_batch_size(self, py_eval_setting):
        if not py_eval_setting.adversarial_attack:
            return self.eval_batch_size
        elif py_eval_setting.adversarial_attack and self.adversarial_batch_size is not None:
            return self.adversarial_batch_size
        return max(self.eval_batch_size // self.ADVERSARIAL_BATCH_SIZE_REDUCTION_FACTOR, 1)


StandardTransform = lambda img_resize_size, img_crop_size: \
    transforms.Compose([
        transforms.Resize(img_resize_size),
        transforms.CenterCrop(img_crop_size),
        transforms.ToTensor(),
    ])


class StandardNormalization():
    def __init__(self, mean, std, input_space='RGB'):
        assert input_space in ['RGB', 'BGR'], \
               f'Can only handle RGB or BGR channel input formats'

        self.mean = mean
        self.std = std
        self.input_space = input_space
