import scipy.special
import timm
import numpy as np
import dill
from functools import partial
import io
import torch
from torch import nn
import sklearn

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb import s3_utils
from mldb.utils import load_model_state_dict, load_model_checkpoint_raw


class TimmFeatures(nn.Module):
    def __init__(self, net):
        super(TimmFeatures, self).__init__()
        self._net = net

    def forward(self, x):
        return self._net.forward_features(x).mean(dim=(2,3))


class IdentityFeatures(nn.Module):
    def __init__(self, net):
        super(IdentityFeatures, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class SklearnCLF(nn.Module):
    def __init__(self, net, clf):
        super(SklearnCLF, self).__init__()
        self.net = net
        self.clf = clf

    def forward(self, x):
        if "Simple1NN" in str(type(self.clf)):
            clf = Simple1NN()
            clf.fit(self.clf.X_train, self.clf.y_train)
            self.clf = clf
        with torch.no_grad():
            x_out = self.net(x)
            if hasattr(self.clf, "decision_function"):
                y_out = scipy.special.softmax(self.clf.decision_function(x_out.cpu().numpy()))
            elif hasattr(self.clf, "predict_proba"):
                y_out = self.clf.predict_proba(x_out.cpu().numpy())
            else:
                assert False
            return torch.Tensor(y_out)


class Simple1NN:
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def decision_function(self, X_test):
        D = self.X_train.dot(X_test.T)
        D *= -2
        D += (np.linalg.norm(self.X_train, axis=1)**2)[: , np.newaxis]
        D += (np.linalg.norm(X_test, axis=1)**2)[np.newaxis, :]
        logits = np.zeros((X_test.shape[0], 1000))
        for i in range(1000):
            logits[:, i] = -1*np.min(D[self.y_train == i, :], axis=0).ravel()
        return logits

    def predict(self, X_test):
        assert False
        # D = self.X_train.dot(X_test.T)
        # D *= -2
        # D += (np.linalg.norm(self.X_train, axis=1)**2)[: , np.newaxis]
        # D += (np.linalg.norm(X_test, axis=1)**2)[np.newaxis, :]
        # return self.y_train[np.argmin(D, axis=0)]


# def resnet50_pretrained_lstsq():
#     # construct featurizer
#     features = TimmFeatures(timm.create_model("resnet50", pretrained=False))
#     # construct clf
#     clf = nn.Linear(2048, 1000, bias=False)
#     net_full = nn.Sequential(features, clf)
#     s3w = s3_utils.S3Wrapper()
#     state_dict = s3w.get("low_accuracy_model_artifacts/resnet50_least_squares.pickle")
#     # load weights
#     net_full.load_state_dict(state_dict)
#     return net_full


def get_dill_loader(model_name):
    def dill_loader():
        return dill.loads(load_model_checkpoint_raw(model_name))
    return dill_loader

def load_resnet(model_name, arch):
    net = timm.create_model(arch)
    load_model_state_dict(net, model_name)
    return net


registry.add_model(
    Model(
        name = 'resnet50_lstsq',
        transform = StandardTransform(img_resize_size=256, img_crop_size=224),
        normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier_loader = get_dill_loader('resnet50_lstsq'),
        eval_batch_size = 256
    )
)

registry.add_model(
    Model(
        name = 'identity32_lstsq',
        transform = StandardTransform(img_resize_size=32, img_crop_size=32),
        normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier_loader = get_dill_loader('identity32_lstsq'),
        eval_batch_size = 256
    )
)

registry.add_model(
    Model(
        name = 'identity32_one_nn',
        transform = StandardTransform(img_resize_size=32, img_crop_size=32),
        normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier_loader = get_dill_loader('identity32_one_nn'),
        eval_batch_size = 256
    )
)

registry.add_model(
    Model(
        name = 'identity32_random_forests',
        transform = StandardTransform(img_resize_size=32, img_crop_size=32),
        normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier_loader = get_dill_loader('identity32_random_forests'),
        eval_batch_size = 256
    )
)

for epoch in range(0, 11):
    registry.add_model(
        Model(
            name = f'resnet101_{epoch}_epochs',
            transform = StandardTransform(img_resize_size=256, img_crop_size=224),
            normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            classifier_loader = partial(load_resnet, f'resnet101_{epoch}_epochs', 'resnet101'),
            eval_batch_size = 256
        )
    )

for epoch in range(1, 100):
    registry.add_model(
        Model(
            name = f'resnet18_50k_{epoch}_epochs',
            transform = StandardTransform(img_resize_size=256, img_crop_size=224),
            normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            classifier_loader = partial(load_resnet, f'resnet18_50k_{epoch}_epochs', 'resnet18'),
            eval_batch_size = 256
        )
    )

for epoch in range(1, 50):
    registry.add_model(
        Model(
            name = f'resnet18_100k_{epoch}_epochs',
            transform = StandardTransform(img_resize_size=256, img_crop_size=224),
            normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            classifier_loader = partial(load_resnet, f'resnet18_100k_{epoch}_epochs', 'resnet18'),
            eval_batch_size = 256
        )
    )

for epoch in range(1, 25):
    registry.add_model(
        Model(
            name = f'resnet18_200k_{epoch}_epochs',
            transform = StandardTransform(img_resize_size=256, img_crop_size=224),
            normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            classifier_loader = partial(load_resnet, f'resnet18_200k_{epoch}_epochs', 'resnet18'),
            eval_batch_size = 256
        )
    )
