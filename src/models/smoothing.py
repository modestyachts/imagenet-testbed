import torch
from scipy.stats import binom_test
import numpy as np
from math import ceil
from torch import nn
import torchvision.models as torch_models

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict
from eval_settings.pgd_attack import pgd_style_attack


model_params = {
'resnet50-randomized_smoothing_noise_0.00': {   'alpha': 1.0,
                                                'arch': 'resnet50',
                                                'eval_batch_size': 256,
                                                'img_crop_size': 224,
                                                'img_resize_size': 256,
                                                'mean': [0.485, 0.456, 0.406],
                                                'n': 100,
                                                'noise_sigma': 0.0,
                                                'std': [0.229, 0.224, 0.225]},
'resnet50-randomized_smoothing_noise_0.25': {   'alpha': 1.0,
                                                'arch': 'resnet50',
                                                'eval_batch_size': 256,
                                                'img_crop_size': 224,
                                                'img_resize_size': 256,
                                                'mean': [0.485, 0.456, 0.406],
                                                'n': 100,
                                                'noise_sigma': 0.25,
                                                'std': [0.229, 0.224, 0.225]},
'resnet50-randomized_smoothing_noise_0.50': {   'alpha': 1.0,
                                                'arch': 'resnet50',
                                                'eval_batch_size': 256,
                                                'img_crop_size': 224,
                                                'img_resize_size': 256,
                                                'mean': [0.485, 0.456, 0.406],
                                                'n': 100,
                                                'noise_sigma': 0.5,
                                                'std': [0.229, 0.224, 0.225]},
'resnet50-randomized_smoothing_noise_1.00': {   'alpha': 1.0,
                                                'arch': 'resnet50',
                                                'eval_batch_size': 256,
                                                'img_crop_size': 224,
                                                'img_resize_size': 256,
                                                'mean': [0.485, 0.456, 0.406],
                                                'n': 100,
                                                'noise_sigma': 1.0,
                                                'std': [0.229, 0.224, 0.225]},
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_0.25': {   'alpha': 1.0,
                                                                    'arch': 'resnet50',
                                                                    'eval_batch_size': 256,
                                                                    'img_crop_size': 224,
                                                                    'img_resize_size': 256,
                                                                    'mean': [0.485, 0.456, 0.406],
                                                                    'n': 100,
                                                                    'noise_sigma': 0.25,
                                                                    'std': [1.0, 1.0, 1.0]},
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_0.50': {   'alpha': 1.0,
                                                                    'arch': 'resnet50',
                                                                    'eval_batch_size': 256,
                                                                    'img_crop_size': 224,
                                                                    'img_resize_size': 256,
                                                                    'mean': [0.485, 0.456, 0.406],
                                                                    'n': 100,
                                                                    'noise_sigma': 0.5,
                                                                    'std': [1.0, 1.0, 1.0]},
'resnet50-smoothing_adversarial_DNN_2steps_eps_512_noise_1.00': {   'alpha': 1.0,
                                                                    'arch': 'resnet50',
                                                                    'eval_batch_size': 256,
                                                                    'img_crop_size': 224,
                                                                    'img_resize_size': 256,
                                                                    'mean': [0.485, 0.456, 0.406],
                                                                    'n': 100,
                                                                    'noise_sigma': 1.0,
                                                                    'std': [1.0, 1.0, 1.0]},
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_0.25': {   'alpha': 1.0,
                                                                   'arch': 'resnet50',
                                                                   'eval_batch_size': 256,
                                                                   'img_crop_size': 224,
                                                                   'img_resize_size': 256,
                                                                   'mean': [0.485, 0.456, 0.406],
                                                                   'n': 100,
                                                                   'noise_sigma': 0.25,
                                                                   'std': [1.0, 1.0, 1.0]},
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_0.50': {   'alpha': 1.0,
                                                                   'arch': 'resnet50',
                                                                   'eval_batch_size': 256,
                                                                   'img_crop_size': 224,
                                                                   'img_resize_size': 256,
                                                                   'mean': [0.485, 0.456, 0.406],
                                                                   'n': 100,
                                                                   'noise_sigma': 0.5,
                                                                   'std': [1.0, 1.0, 1.0]},
'resnet50-smoothing_adversarial_PGD_1step_eps_512_noise_1.00': {   'alpha': 1.0,
                                                                   'arch': 'resnet50',
                                                                   'eval_batch_size': 256,
                                                                   'img_crop_size': 224,
                                                                   'img_resize_size': 256,
                                                                   'mean': [0.485, 0.456, 0.406],
                                                                   'n': 100,
                                                                   'noise_sigma': 1.0,
                                                                   'std': [1.0, 1.0, 1.0]}}


# THIS SECTION OF THE CODE IS TAKEN FROM https://github.com/locuslab/smoothing

class Smooth(nn.Module):
    """A smoothed classifier g """

    def __init__(self, base_classifier, sigma, n, alpha, mean, std):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param sigma: the noise level hyperparameter
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.n = n
        self.alpha = alpha
        self.mean = nn.Parameter(torch.tensor(mean).float().view(3, 1, 1))
        self.std = nn.Parameter(torch.tensor(std).float().view(3, 1, 1))

    def predict(self, x, batch_size, class_sublist=None):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        counts = self._sample_noise(x, self.n, batch_size, class_sublist)
        top2 = counts.argsort()[::-1]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > self.alpha:
            return Smooth.ABSTAIN
        else:
            return counts

    def _sample_noise(self, x, num, batch_size, class_sublist=None):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = []
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                logits = self.base_classify(batch + noise)

                if class_sublist is not None:
                    logits = logits.t()[class_sublist].t()
                predictions = logits.argmax(dim=1).cpu().numpy()

                counts += [self._count_arr(predictions, logits.size(1))]
            return np.array(counts).sum(axis=0)

    def _count_arr(self, arr, length):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def predict_batch(self, x, class_sublist):
        counts = []
        for img in x:
            count = self.predict(img, x.size(0), class_sublist)
            counts += [torch.from_numpy(count)]
        counts = torch.stack(counts, dim=0)
        return counts.float()

    def forward(self, x):
        """ Definition for forward pass during adversarial (pgd) attack.
        Not meant to be the main form of evaluation. For that, see predict_batch.
        """
        noise = torch.randn_like(x) * self.sigma
        return self.base_classify(x + noise)

    def base_classify(self, x):
        x = (x - self.mean) / self.std
        return self.base_classifier(x)

# END OF ATTRIBUTED SECTION


def gen_classifier_loader(name, d):
    def classifier_loader():
        model = torch_models.__dict__[d['arch']]()
        load_model_state_dict(model, name)
        model = Smooth(model, d['noise_sigma'], d['n'], d['alpha'], d['mean'], d['std'])
        return model
    return classifier_loader


def classify(images, model, class_sublist, adversarial_attack):
    if adversarial_attack:
        images = pgd_style_attack(adversarial_attack, images, model)
    return model.predict_batch(images, class_sublist=class_sublist)


for name, d in model_params.items():
    registry.add_model(
        Model(
            name = name,
            arch = d['arch'],
            transform = StandardTransform(d['img_resize_size'], d['img_crop_size']),
            classifier_loader = gen_classifier_loader(name, d),
            eval_batch_size = d['eval_batch_size'],
            adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None,
            classify = classify,
        )
    )
