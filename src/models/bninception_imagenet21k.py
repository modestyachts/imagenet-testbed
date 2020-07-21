import pathlib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_checkpoint_bytes


__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv_conv1 = self.__conv(2, name='conv_conv1', in_channels=3, out_channels=96, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.bn_conv1 = self.__batch_normalization(2, 'bn_conv1', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_conv2red = self.__conv(2, name='conv_conv2red', in_channels=96, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_conv2red = self.__batch_normalization(2, 'bn_conv2red', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_conv2 = self.__conv(2, name='conv_conv2', in_channels=128, out_channels=288, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_conv2 = self.__batch_normalization(2, 'bn_conv2', num_features=288, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3a_1x1 = self.__conv(2, name='conv_3a_1x1', in_channels=288, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_3a_3x3_reduce = self.__conv(2, name='conv_3a_3x3_reduce', in_channels=288, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_3a_double_3x3_reduce = self.__conv(2, name='conv_3a_double_3x3_reduce', in_channels=288, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_3a_1x1 = self.__batch_normalization(2, 'bn_3a_1x1', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_3a_3x3_reduce = self.__batch_normalization(2, 'bn_3a_3x3_reduce', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_3a_double_3x3_reduce = self.__batch_normalization(2, 'bn_3a_double_3x3_reduce', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3a_proj = self.__conv(2, name='conv_3a_proj', in_channels=288, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_3a_proj = self.__batch_normalization(2, 'bn_3a_proj', num_features=48, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3a_3x3 = self.__conv(2, name='conv_3a_3x3', in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv_3a_double_3x3_0 = self.__conv(2, name='conv_3a_double_3x3_0', in_channels=96, out_channels=144, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_3a_3x3 = self.__batch_normalization(2, 'bn_3a_3x3', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_3a_double_3x3_0 = self.__batch_normalization(2, 'bn_3a_double_3x3_0', num_features=144, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3a_double_3x3_1 = self.__conv(2, name='conv_3a_double_3x3_1', in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_3a_double_3x3_1 = self.__batch_normalization(2, 'bn_3a_double_3x3_1', num_features=144, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3b_1x1 = self.__conv(2, name='conv_3b_1x1', in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_3b_3x3_reduce = self.__conv(2, name='conv_3b_3x3_reduce', in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_3b_double_3x3_reduce = self.__conv(2, name='conv_3b_double_3x3_reduce', in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_3b_1x1 = self.__batch_normalization(2, 'bn_3b_1x1', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_3b_3x3_reduce = self.__batch_normalization(2, 'bn_3b_3x3_reduce', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_3b_double_3x3_reduce = self.__batch_normalization(2, 'bn_3b_double_3x3_reduce', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3b_proj = self.__conv(2, name='conv_3b_proj', in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_3b_proj = self.__batch_normalization(2, 'bn_3b_proj', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3b_3x3 = self.__conv(2, name='conv_3b_3x3', in_channels=96, out_channels=144, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv_3b_double_3x3_0 = self.__conv(2, name='conv_3b_double_3x3_0', in_channels=96, out_channels=144, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_3b_3x3 = self.__batch_normalization(2, 'bn_3b_3x3', num_features=144, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_3b_double_3x3_0 = self.__batch_normalization(2, 'bn_3b_double_3x3_0', num_features=144, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3b_double_3x3_1 = self.__conv(2, name='conv_3b_double_3x3_1', in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_3b_double_3x3_1 = self.__batch_normalization(2, 'bn_3b_double_3x3_1', num_features=144, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3c_3x3_reduce = self.__conv(2, name='conv_3c_3x3_reduce', in_channels=480, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_3c_double_3x3_reduce = self.__conv(2, name='conv_3c_double_3x3_reduce', in_channels=480, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_3c_3x3_reduce = self.__batch_normalization(2, 'bn_3c_3x3_reduce', num_features=192, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_3c_double_3x3_reduce = self.__batch_normalization(2, 'bn_3c_double_3x3_reduce', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3c_3x3 = self.__conv(2, name='conv_3c_3x3', in_channels=192, out_channels=240, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True)
        self.conv_3c_double_3x3_0 = self.__conv(2, name='conv_3c_double_3x3_0', in_channels=96, out_channels=144, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_3c_3x3 = self.__batch_normalization(2, 'bn_3c_3x3', num_features=240, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_3c_double_3x3_0 = self.__batch_normalization(2, 'bn_3c_double_3x3_0', num_features=144, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_3c_double_3x3_1 = self.__conv(2, name='conv_3c_double_3x3_1', in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True)
        self.bn_3c_double_3x3_1 = self.__batch_normalization(2, 'bn_3c_double_3x3_1', num_features=144, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4a_1x1 = self.__conv(2, name='conv_4a_1x1', in_channels=864, out_channels=224, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4a_3x3_reduce = self.__conv(2, name='conv_4a_3x3_reduce', in_channels=864, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4a_double_3x3_reduce = self.__conv(2, name='conv_4a_double_3x3_reduce', in_channels=864, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4a_1x1 = self.__batch_normalization(2, 'bn_4a_1x1', num_features=224, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4a_3x3_reduce = self.__batch_normalization(2, 'bn_4a_3x3_reduce', num_features=64, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4a_double_3x3_reduce = self.__batch_normalization(2, 'bn_4a_double_3x3_reduce', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4a_proj = self.__conv(2, name='conv_4a_proj', in_channels=864, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4a_proj = self.__batch_normalization(2, 'bn_4a_proj', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4a_3x3 = self.__conv(2, name='conv_4a_3x3', in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv_4a_double_3x3_0 = self.__conv(2, name='conv_4a_double_3x3_0', in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4a_3x3 = self.__batch_normalization(2, 'bn_4a_3x3', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4a_double_3x3_0 = self.__batch_normalization(2, 'bn_4a_double_3x3_0', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4a_double_3x3_1 = self.__conv(2, name='conv_4a_double_3x3_1', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4a_double_3x3_1 = self.__batch_normalization(2, 'bn_4a_double_3x3_1', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4b_1x1 = self.__conv(2, name='conv_4b_1x1', in_channels=576, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4b_3x3_reduce = self.__conv(2, name='conv_4b_3x3_reduce', in_channels=576, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4b_double_3x3_reduce = self.__conv(2, name='conv_4b_double_3x3_reduce', in_channels=576, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4b_1x1 = self.__batch_normalization(2, 'bn_4b_1x1', num_features=192, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4b_3x3_reduce = self.__batch_normalization(2, 'bn_4b_3x3_reduce', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4b_double_3x3_reduce = self.__batch_normalization(2, 'bn_4b_double_3x3_reduce', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4b_proj = self.__conv(2, name='conv_4b_proj', in_channels=576, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4b_proj = self.__batch_normalization(2, 'bn_4b_proj', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4b_3x3 = self.__conv(2, name='conv_4b_3x3', in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv_4b_double_3x3_0 = self.__conv(2, name='conv_4b_double_3x3_0', in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4b_3x3 = self.__batch_normalization(2, 'bn_4b_3x3', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4b_double_3x3_0 = self.__batch_normalization(2, 'bn_4b_double_3x3_0', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4b_double_3x3_1 = self.__conv(2, name='conv_4b_double_3x3_1', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4b_double_3x3_1 = self.__batch_normalization(2, 'bn_4b_double_3x3_1', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4c_1x1 = self.__conv(2, name='conv_4c_1x1', in_channels=576, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4c_3x3_reduce = self.__conv(2, name='conv_4c_3x3_reduce', in_channels=576, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4c_double_3x3_reduce = self.__conv(2, name='conv_4c_double_3x3_reduce', in_channels=576, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4c_1x1 = self.__batch_normalization(2, 'bn_4c_1x1', num_features=160, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4c_3x3_reduce = self.__batch_normalization(2, 'bn_4c_3x3_reduce', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4c_double_3x3_reduce = self.__batch_normalization(2, 'bn_4c_double_3x3_reduce', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4c_proj = self.__conv(2, name='conv_4c_proj', in_channels=576, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4c_proj = self.__batch_normalization(2, 'bn_4c_proj', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4c_3x3 = self.__conv(2, name='conv_4c_3x3', in_channels=128, out_channels=160, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv_4c_double_3x3_0 = self.__conv(2, name='conv_4c_double_3x3_0', in_channels=128, out_channels=160, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4c_3x3 = self.__batch_normalization(2, 'bn_4c_3x3', num_features=160, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4c_double_3x3_0 = self.__batch_normalization(2, 'bn_4c_double_3x3_0', num_features=160, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4c_double_3x3_1 = self.__conv(2, name='conv_4c_double_3x3_1', in_channels=160, out_channels=160, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4c_double_3x3_1 = self.__batch_normalization(2, 'bn_4c_double_3x3_1', num_features=160, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4d_1x1 = self.__conv(2, name='conv_4d_1x1', in_channels=608, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4d_3x3_reduce = self.__conv(2, name='conv_4d_3x3_reduce', in_channels=608, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4d_double_3x3_reduce = self.__conv(2, name='conv_4d_double_3x3_reduce', in_channels=608, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4d_1x1 = self.__batch_normalization(2, 'bn_4d_1x1', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4d_3x3_reduce = self.__batch_normalization(2, 'bn_4d_3x3_reduce', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4d_double_3x3_reduce = self.__batch_normalization(2, 'bn_4d_double_3x3_reduce', num_features=160, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4d_proj = self.__conv(2, name='conv_4d_proj', in_channels=608, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4d_proj = self.__batch_normalization(2, 'bn_4d_proj', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4d_3x3 = self.__conv(2, name='conv_4d_3x3', in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv_4d_double_3x3_0 = self.__conv(2, name='conv_4d_double_3x3_0', in_channels=160, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4d_3x3 = self.__batch_normalization(2, 'bn_4d_3x3', num_features=192, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4d_double_3x3_0 = self.__batch_normalization(2, 'bn_4d_double_3x3_0', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4d_double_3x3_1 = self.__conv(2, name='conv_4d_double_3x3_1', in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4d_double_3x3_1 = self.__batch_normalization(2, 'bn_4d_double_3x3_1', num_features=96, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4e_3x3_reduce = self.__conv(2, name='conv_4e_3x3_reduce', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_4e_double_3x3_reduce = self.__conv(2, name='conv_4e_double_3x3_reduce', in_channels=512, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_4e_3x3_reduce = self.__batch_normalization(2, 'bn_4e_3x3_reduce', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4e_double_3x3_reduce = self.__batch_normalization(2, 'bn_4e_double_3x3_reduce', num_features=192, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4e_3x3 = self.__conv(2, name='conv_4e_3x3', in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True)
        self.conv_4e_double_3x3_0 = self.__conv(2, name='conv_4e_double_3x3_0', in_channels=192, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_4e_3x3 = self.__batch_normalization(2, 'bn_4e_3x3', num_features=192, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_4e_double_3x3_0 = self.__batch_normalization(2, 'bn_4e_double_3x3_0', num_features=256, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_4e_double_3x3_1 = self.__conv(2, name='conv_4e_double_3x3_1', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True)
        self.bn_4e_double_3x3_1 = self.__batch_normalization(2, 'bn_4e_double_3x3_1', num_features=256, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_5a_1x1 = self.__conv(2, name='conv_5a_1x1', in_channels=960, out_channels=352, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_5a_3x3_reduce = self.__conv(2, name='conv_5a_3x3_reduce', in_channels=960, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_5a_double_3x3_reduce = self.__conv(2, name='conv_5a_double_3x3_reduce', in_channels=960, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_5a_1x1 = self.__batch_normalization(2, 'bn_5a_1x1', num_features=352, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_5a_3x3_reduce = self.__batch_normalization(2, 'bn_5a_3x3_reduce', num_features=192, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_5a_double_3x3_reduce = self.__batch_normalization(2, 'bn_5a_double_3x3_reduce', num_features=160, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_5a_proj = self.__conv(2, name='conv_5a_proj', in_channels=960, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_5a_proj = self.__batch_normalization(2, 'bn_5a_proj', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_5a_3x3 = self.__conv(2, name='conv_5a_3x3', in_channels=192, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv_5a_double_3x3_0 = self.__conv(2, name='conv_5a_double_3x3_0', in_channels=160, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_5a_3x3 = self.__batch_normalization(2, 'bn_5a_3x3', num_features=320, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_5a_double_3x3_0 = self.__batch_normalization(2, 'bn_5a_double_3x3_0', num_features=224, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_5a_double_3x3_1 = self.__conv(2, name='conv_5a_double_3x3_1', in_channels=224, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_5a_double_3x3_1 = self.__batch_normalization(2, 'bn_5a_double_3x3_1', num_features=224, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_5b_1x1 = self.__conv(2, name='conv_5b_1x1', in_channels=1024, out_channels=352, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_5b_3x3_reduce = self.__conv(2, name='conv_5b_3x3_reduce', in_channels=1024, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv_5b_double_3x3_reduce = self.__conv(2, name='conv_5b_double_3x3_reduce', in_channels=1024, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_5b_1x1 = self.__batch_normalization(2, 'bn_5b_1x1', num_features=352, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_5b_3x3_reduce = self.__batch_normalization(2, 'bn_5b_3x3_reduce', num_features=192, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_5b_double_3x3_reduce = self.__batch_normalization(2, 'bn_5b_double_3x3_reduce', num_features=192, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_5b_proj = self.__conv(2, name='conv_5b_proj', in_channels=1024, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.bn_5b_proj = self.__batch_normalization(2, 'bn_5b_proj', num_features=128, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_5b_3x3 = self.__conv(2, name='conv_5b_3x3', in_channels=192, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv_5b_double_3x3_0 = self.__conv(2, name='conv_5b_double_3x3_0', in_channels=192, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_5b_3x3 = self.__batch_normalization(2, 'bn_5b_3x3', num_features=320, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.bn_5b_double_3x3_0 = self.__batch_normalization(2, 'bn_5b_double_3x3_0', num_features=224, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.conv_5b_double_3x3_1 = self.__conv(2, name='conv_5b_double_3x3_1', in_channels=224, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.bn_5b_double_3x3_1 = self.__batch_normalization(2, 'bn_5b_double_3x3_1', num_features=224, eps=9.999999747378752e-05, momentum=0.8999999761581421)
        self.fc1 = self.__dense(name = 'fc1', in_features = 1024, out_features = 21841, bias = True)

    def forward(self, x):
        conv_conv1_pad  = F.pad(x, (3, 3, 3, 3))
        conv_conv1      = self.conv_conv1(conv_conv1_pad)
        bn_conv1        = self.bn_conv1(conv_conv1)
        relu_conv1      = F.relu(bn_conv1)
        pool1           = F.max_pool2d(relu_conv1, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        conv_conv2red   = self.conv_conv2red(pool1)
        bn_conv2red     = self.bn_conv2red(conv_conv2red)
        relu_conv2red   = F.relu(bn_conv2red)
        conv_conv2_pad  = F.pad(relu_conv2red, (1, 1, 1, 1))
        conv_conv2      = self.conv_conv2(conv_conv2_pad)
        bn_conv2        = self.bn_conv2(conv_conv2)
        relu_conv2      = F.relu(bn_conv2)
        pool2           = F.max_pool2d(relu_conv2, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        conv_3a_1x1     = self.conv_3a_1x1(pool2)
        conv_3a_3x3_reduce = self.conv_3a_3x3_reduce(pool2)
        conv_3a_double_3x3_reduce = self.conv_3a_double_3x3_reduce(pool2)
        avg_pool_3a_pool = F.avg_pool2d(pool2, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        bn_3a_1x1       = self.bn_3a_1x1(conv_3a_1x1)
        bn_3a_3x3_reduce = self.bn_3a_3x3_reduce(conv_3a_3x3_reduce)
        bn_3a_double_3x3_reduce = self.bn_3a_double_3x3_reduce(conv_3a_double_3x3_reduce)
        conv_3a_proj    = self.conv_3a_proj(avg_pool_3a_pool)
        relu_3a_1x1     = F.relu(bn_3a_1x1)
        relu_3a_3x3_reduce = F.relu(bn_3a_3x3_reduce)
        relu_3a_double_3x3_reduce = F.relu(bn_3a_double_3x3_reduce)
        bn_3a_proj      = self.bn_3a_proj(conv_3a_proj)
        conv_3a_3x3_pad = F.pad(relu_3a_3x3_reduce, (1, 1, 1, 1))
        conv_3a_3x3     = self.conv_3a_3x3(conv_3a_3x3_pad)
        conv_3a_double_3x3_0_pad = F.pad(relu_3a_double_3x3_reduce, (1, 1, 1, 1))
        conv_3a_double_3x3_0 = self.conv_3a_double_3x3_0(conv_3a_double_3x3_0_pad)
        relu_3a_proj    = F.relu(bn_3a_proj)
        bn_3a_3x3       = self.bn_3a_3x3(conv_3a_3x3)
        bn_3a_double_3x3_0 = self.bn_3a_double_3x3_0(conv_3a_double_3x3_0)
        relu_3a_3x3     = F.relu(bn_3a_3x3)
        relu_3a_double_3x3_0 = F.relu(bn_3a_double_3x3_0)
        conv_3a_double_3x3_1_pad = F.pad(relu_3a_double_3x3_0, (1, 1, 1, 1))
        conv_3a_double_3x3_1 = self.conv_3a_double_3x3_1(conv_3a_double_3x3_1_pad)
        bn_3a_double_3x3_1 = self.bn_3a_double_3x3_1(conv_3a_double_3x3_1)
        relu_3a_double_3x3_1 = F.relu(bn_3a_double_3x3_1)
        ch_concat_3a_chconcat = torch.cat((relu_3a_1x1, relu_3a_3x3, relu_3a_double_3x3_1, relu_3a_proj), 1)
        conv_3b_1x1     = self.conv_3b_1x1(ch_concat_3a_chconcat)
        conv_3b_3x3_reduce = self.conv_3b_3x3_reduce(ch_concat_3a_chconcat)
        conv_3b_double_3x3_reduce = self.conv_3b_double_3x3_reduce(ch_concat_3a_chconcat)
        avg_pool_3b_pool = F.avg_pool2d(ch_concat_3a_chconcat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        bn_3b_1x1       = self.bn_3b_1x1(conv_3b_1x1)
        bn_3b_3x3_reduce = self.bn_3b_3x3_reduce(conv_3b_3x3_reduce)
        bn_3b_double_3x3_reduce = self.bn_3b_double_3x3_reduce(conv_3b_double_3x3_reduce)
        conv_3b_proj    = self.conv_3b_proj(avg_pool_3b_pool)
        relu_3b_1x1     = F.relu(bn_3b_1x1)
        relu_3b_3x3_reduce = F.relu(bn_3b_3x3_reduce)
        relu_3b_double_3x3_reduce = F.relu(bn_3b_double_3x3_reduce)
        bn_3b_proj      = self.bn_3b_proj(conv_3b_proj)
        conv_3b_3x3_pad = F.pad(relu_3b_3x3_reduce, (1, 1, 1, 1))
        conv_3b_3x3     = self.conv_3b_3x3(conv_3b_3x3_pad)
        conv_3b_double_3x3_0_pad = F.pad(relu_3b_double_3x3_reduce, (1, 1, 1, 1))
        conv_3b_double_3x3_0 = self.conv_3b_double_3x3_0(conv_3b_double_3x3_0_pad)
        relu_3b_proj    = F.relu(bn_3b_proj)
        bn_3b_3x3       = self.bn_3b_3x3(conv_3b_3x3)
        bn_3b_double_3x3_0 = self.bn_3b_double_3x3_0(conv_3b_double_3x3_0)
        relu_3b_3x3     = F.relu(bn_3b_3x3)
        relu_3b_double_3x3_0 = F.relu(bn_3b_double_3x3_0)
        conv_3b_double_3x3_1_pad = F.pad(relu_3b_double_3x3_0, (1, 1, 1, 1))
        conv_3b_double_3x3_1 = self.conv_3b_double_3x3_1(conv_3b_double_3x3_1_pad)
        bn_3b_double_3x3_1 = self.bn_3b_double_3x3_1(conv_3b_double_3x3_1)
        relu_3b_double_3x3_1 = F.relu(bn_3b_double_3x3_1)
        ch_concat_3b_chconcat = torch.cat((relu_3b_1x1, relu_3b_3x3, relu_3b_double_3x3_1, relu_3b_proj), 1)
        conv_3c_3x3_reduce = self.conv_3c_3x3_reduce(ch_concat_3b_chconcat)
        conv_3c_double_3x3_reduce = self.conv_3c_double_3x3_reduce(ch_concat_3b_chconcat)
        max_pool_3c_pool_pad = F.pad(ch_concat_3b_chconcat, (1, 1, 1, 1), value=float('-inf'))
        max_pool_3c_pool = F.max_pool2d(max_pool_3c_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        bn_3c_3x3_reduce = self.bn_3c_3x3_reduce(conv_3c_3x3_reduce)
        bn_3c_double_3x3_reduce = self.bn_3c_double_3x3_reduce(conv_3c_double_3x3_reduce)
        relu_3c_3x3_reduce = F.relu(bn_3c_3x3_reduce)
        relu_3c_double_3x3_reduce = F.relu(bn_3c_double_3x3_reduce)
        conv_3c_3x3_pad = F.pad(relu_3c_3x3_reduce, (1, 1, 1, 1))
        conv_3c_3x3     = self.conv_3c_3x3(conv_3c_3x3_pad)
        conv_3c_double_3x3_0_pad = F.pad(relu_3c_double_3x3_reduce, (1, 1, 1, 1))
        conv_3c_double_3x3_0 = self.conv_3c_double_3x3_0(conv_3c_double_3x3_0_pad)
        bn_3c_3x3       = self.bn_3c_3x3(conv_3c_3x3)
        bn_3c_double_3x3_0 = self.bn_3c_double_3x3_0(conv_3c_double_3x3_0)
        relu_3c_3x3     = F.relu(bn_3c_3x3)
        relu_3c_double_3x3_0 = F.relu(bn_3c_double_3x3_0)
        conv_3c_double_3x3_1_pad = F.pad(relu_3c_double_3x3_0, (1, 1, 1, 1))
        conv_3c_double_3x3_1 = self.conv_3c_double_3x3_1(conv_3c_double_3x3_1_pad)
        bn_3c_double_3x3_1 = self.bn_3c_double_3x3_1(conv_3c_double_3x3_1)
        relu_3c_double_3x3_1 = F.relu(bn_3c_double_3x3_1)
        ch_concat_3c_chconcat = torch.cat((relu_3c_3x3, relu_3c_double_3x3_1, max_pool_3c_pool), 1)
        conv_4a_1x1     = self.conv_4a_1x1(ch_concat_3c_chconcat)
        conv_4a_3x3_reduce = self.conv_4a_3x3_reduce(ch_concat_3c_chconcat)
        conv_4a_double_3x3_reduce = self.conv_4a_double_3x3_reduce(ch_concat_3c_chconcat)
        avg_pool_4a_pool = F.avg_pool2d(ch_concat_3c_chconcat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        bn_4a_1x1       = self.bn_4a_1x1(conv_4a_1x1)
        bn_4a_3x3_reduce = self.bn_4a_3x3_reduce(conv_4a_3x3_reduce)
        bn_4a_double_3x3_reduce = self.bn_4a_double_3x3_reduce(conv_4a_double_3x3_reduce)
        conv_4a_proj    = self.conv_4a_proj(avg_pool_4a_pool)
        relu_4a_1x1     = F.relu(bn_4a_1x1)
        relu_4a_3x3_reduce = F.relu(bn_4a_3x3_reduce)
        relu_4a_double_3x3_reduce = F.relu(bn_4a_double_3x3_reduce)
        bn_4a_proj      = self.bn_4a_proj(conv_4a_proj)
        conv_4a_3x3_pad = F.pad(relu_4a_3x3_reduce, (1, 1, 1, 1))
        conv_4a_3x3     = self.conv_4a_3x3(conv_4a_3x3_pad)
        conv_4a_double_3x3_0_pad = F.pad(relu_4a_double_3x3_reduce, (1, 1, 1, 1))
        conv_4a_double_3x3_0 = self.conv_4a_double_3x3_0(conv_4a_double_3x3_0_pad)
        relu_4a_proj    = F.relu(bn_4a_proj)
        bn_4a_3x3       = self.bn_4a_3x3(conv_4a_3x3)
        bn_4a_double_3x3_0 = self.bn_4a_double_3x3_0(conv_4a_double_3x3_0)
        relu_4a_3x3     = F.relu(bn_4a_3x3)
        relu_4a_double_3x3_0 = F.relu(bn_4a_double_3x3_0)
        conv_4a_double_3x3_1_pad = F.pad(relu_4a_double_3x3_0, (1, 1, 1, 1))
        conv_4a_double_3x3_1 = self.conv_4a_double_3x3_1(conv_4a_double_3x3_1_pad)
        bn_4a_double_3x3_1 = self.bn_4a_double_3x3_1(conv_4a_double_3x3_1)
        relu_4a_double_3x3_1 = F.relu(bn_4a_double_3x3_1)
        ch_concat_4a_chconcat = torch.cat((relu_4a_1x1, relu_4a_3x3, relu_4a_double_3x3_1, relu_4a_proj), 1)
        conv_4b_1x1     = self.conv_4b_1x1(ch_concat_4a_chconcat)
        conv_4b_3x3_reduce = self.conv_4b_3x3_reduce(ch_concat_4a_chconcat)
        conv_4b_double_3x3_reduce = self.conv_4b_double_3x3_reduce(ch_concat_4a_chconcat)
        avg_pool_4b_pool = F.avg_pool2d(ch_concat_4a_chconcat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        bn_4b_1x1       = self.bn_4b_1x1(conv_4b_1x1)
        bn_4b_3x3_reduce = self.bn_4b_3x3_reduce(conv_4b_3x3_reduce)
        bn_4b_double_3x3_reduce = self.bn_4b_double_3x3_reduce(conv_4b_double_3x3_reduce)
        conv_4b_proj    = self.conv_4b_proj(avg_pool_4b_pool)
        relu_4b_1x1     = F.relu(bn_4b_1x1)
        relu_4b_3x3_reduce = F.relu(bn_4b_3x3_reduce)
        relu_4b_double_3x3_reduce = F.relu(bn_4b_double_3x3_reduce)
        bn_4b_proj      = self.bn_4b_proj(conv_4b_proj)
        conv_4b_3x3_pad = F.pad(relu_4b_3x3_reduce, (1, 1, 1, 1))
        conv_4b_3x3     = self.conv_4b_3x3(conv_4b_3x3_pad)
        conv_4b_double_3x3_0_pad = F.pad(relu_4b_double_3x3_reduce, (1, 1, 1, 1))
        conv_4b_double_3x3_0 = self.conv_4b_double_3x3_0(conv_4b_double_3x3_0_pad)
        relu_4b_proj    = F.relu(bn_4b_proj)
        bn_4b_3x3       = self.bn_4b_3x3(conv_4b_3x3)
        bn_4b_double_3x3_0 = self.bn_4b_double_3x3_0(conv_4b_double_3x3_0)
        relu_4b_3x3     = F.relu(bn_4b_3x3)
        relu_4b_double_3x3_0 = F.relu(bn_4b_double_3x3_0)
        conv_4b_double_3x3_1_pad = F.pad(relu_4b_double_3x3_0, (1, 1, 1, 1))
        conv_4b_double_3x3_1 = self.conv_4b_double_3x3_1(conv_4b_double_3x3_1_pad)
        bn_4b_double_3x3_1 = self.bn_4b_double_3x3_1(conv_4b_double_3x3_1)
        relu_4b_double_3x3_1 = F.relu(bn_4b_double_3x3_1)
        ch_concat_4b_chconcat = torch.cat((relu_4b_1x1, relu_4b_3x3, relu_4b_double_3x3_1, relu_4b_proj), 1)
        conv_4c_1x1     = self.conv_4c_1x1(ch_concat_4b_chconcat)
        conv_4c_3x3_reduce = self.conv_4c_3x3_reduce(ch_concat_4b_chconcat)
        conv_4c_double_3x3_reduce = self.conv_4c_double_3x3_reduce(ch_concat_4b_chconcat)
        avg_pool_4c_pool = F.avg_pool2d(ch_concat_4b_chconcat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        bn_4c_1x1       = self.bn_4c_1x1(conv_4c_1x1)
        bn_4c_3x3_reduce = self.bn_4c_3x3_reduce(conv_4c_3x3_reduce)
        bn_4c_double_3x3_reduce = self.bn_4c_double_3x3_reduce(conv_4c_double_3x3_reduce)
        conv_4c_proj    = self.conv_4c_proj(avg_pool_4c_pool)
        relu_4c_1x1     = F.relu(bn_4c_1x1)
        relu_4c_3x3_reduce = F.relu(bn_4c_3x3_reduce)
        relu_4c_double_3x3_reduce = F.relu(bn_4c_double_3x3_reduce)
        bn_4c_proj      = self.bn_4c_proj(conv_4c_proj)
        conv_4c_3x3_pad = F.pad(relu_4c_3x3_reduce, (1, 1, 1, 1))
        conv_4c_3x3     = self.conv_4c_3x3(conv_4c_3x3_pad)
        conv_4c_double_3x3_0_pad = F.pad(relu_4c_double_3x3_reduce, (1, 1, 1, 1))
        conv_4c_double_3x3_0 = self.conv_4c_double_3x3_0(conv_4c_double_3x3_0_pad)
        relu_4c_proj    = F.relu(bn_4c_proj)
        bn_4c_3x3       = self.bn_4c_3x3(conv_4c_3x3)
        bn_4c_double_3x3_0 = self.bn_4c_double_3x3_0(conv_4c_double_3x3_0)
        relu_4c_3x3     = F.relu(bn_4c_3x3)
        relu_4c_double_3x3_0 = F.relu(bn_4c_double_3x3_0)
        conv_4c_double_3x3_1_pad = F.pad(relu_4c_double_3x3_0, (1, 1, 1, 1))
        conv_4c_double_3x3_1 = self.conv_4c_double_3x3_1(conv_4c_double_3x3_1_pad)
        bn_4c_double_3x3_1 = self.bn_4c_double_3x3_1(conv_4c_double_3x3_1)
        relu_4c_double_3x3_1 = F.relu(bn_4c_double_3x3_1)
        ch_concat_4c_chconcat = torch.cat((relu_4c_1x1, relu_4c_3x3, relu_4c_double_3x3_1, relu_4c_proj), 1)
        conv_4d_1x1     = self.conv_4d_1x1(ch_concat_4c_chconcat)
        conv_4d_3x3_reduce = self.conv_4d_3x3_reduce(ch_concat_4c_chconcat)
        conv_4d_double_3x3_reduce = self.conv_4d_double_3x3_reduce(ch_concat_4c_chconcat)
        avg_pool_4d_pool = F.avg_pool2d(ch_concat_4c_chconcat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        bn_4d_1x1       = self.bn_4d_1x1(conv_4d_1x1)
        bn_4d_3x3_reduce = self.bn_4d_3x3_reduce(conv_4d_3x3_reduce)
        bn_4d_double_3x3_reduce = self.bn_4d_double_3x3_reduce(conv_4d_double_3x3_reduce)
        conv_4d_proj    = self.conv_4d_proj(avg_pool_4d_pool)
        relu_4d_1x1     = F.relu(bn_4d_1x1)
        relu_4d_3x3_reduce = F.relu(bn_4d_3x3_reduce)
        relu_4d_double_3x3_reduce = F.relu(bn_4d_double_3x3_reduce)
        bn_4d_proj      = self.bn_4d_proj(conv_4d_proj)
        conv_4d_3x3_pad = F.pad(relu_4d_3x3_reduce, (1, 1, 1, 1))
        conv_4d_3x3     = self.conv_4d_3x3(conv_4d_3x3_pad)
        conv_4d_double_3x3_0_pad = F.pad(relu_4d_double_3x3_reduce, (1, 1, 1, 1))
        conv_4d_double_3x3_0 = self.conv_4d_double_3x3_0(conv_4d_double_3x3_0_pad)
        relu_4d_proj    = F.relu(bn_4d_proj)
        bn_4d_3x3       = self.bn_4d_3x3(conv_4d_3x3)
        bn_4d_double_3x3_0 = self.bn_4d_double_3x3_0(conv_4d_double_3x3_0)
        relu_4d_3x3     = F.relu(bn_4d_3x3)
        relu_4d_double_3x3_0 = F.relu(bn_4d_double_3x3_0)
        conv_4d_double_3x3_1_pad = F.pad(relu_4d_double_3x3_0, (1, 1, 1, 1))
        conv_4d_double_3x3_1 = self.conv_4d_double_3x3_1(conv_4d_double_3x3_1_pad)
        bn_4d_double_3x3_1 = self.bn_4d_double_3x3_1(conv_4d_double_3x3_1)
        relu_4d_double_3x3_1 = F.relu(bn_4d_double_3x3_1)
        ch_concat_4d_chconcat = torch.cat((relu_4d_1x1, relu_4d_3x3, relu_4d_double_3x3_1, relu_4d_proj), 1)
        conv_4e_3x3_reduce = self.conv_4e_3x3_reduce(ch_concat_4d_chconcat)
        conv_4e_double_3x3_reduce = self.conv_4e_double_3x3_reduce(ch_concat_4d_chconcat)
        max_pool_4e_pool_pad = F.pad(ch_concat_4d_chconcat, (1, 1, 1, 1), value=float('-inf'))
        max_pool_4e_pool = F.max_pool2d(max_pool_4e_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        bn_4e_3x3_reduce = self.bn_4e_3x3_reduce(conv_4e_3x3_reduce)
        bn_4e_double_3x3_reduce = self.bn_4e_double_3x3_reduce(conv_4e_double_3x3_reduce)
        relu_4e_3x3_reduce = F.relu(bn_4e_3x3_reduce)
        relu_4e_double_3x3_reduce = F.relu(bn_4e_double_3x3_reduce)
        conv_4e_3x3_pad = F.pad(relu_4e_3x3_reduce, (1, 1, 1, 1))
        conv_4e_3x3     = self.conv_4e_3x3(conv_4e_3x3_pad)
        conv_4e_double_3x3_0_pad = F.pad(relu_4e_double_3x3_reduce, (1, 1, 1, 1))
        conv_4e_double_3x3_0 = self.conv_4e_double_3x3_0(conv_4e_double_3x3_0_pad)
        bn_4e_3x3       = self.bn_4e_3x3(conv_4e_3x3)
        bn_4e_double_3x3_0 = self.bn_4e_double_3x3_0(conv_4e_double_3x3_0)
        relu_4e_3x3     = F.relu(bn_4e_3x3)
        relu_4e_double_3x3_0 = F.relu(bn_4e_double_3x3_0)
        conv_4e_double_3x3_1_pad = F.pad(relu_4e_double_3x3_0, (1, 1, 1, 1))
        conv_4e_double_3x3_1 = self.conv_4e_double_3x3_1(conv_4e_double_3x3_1_pad)
        bn_4e_double_3x3_1 = self.bn_4e_double_3x3_1(conv_4e_double_3x3_1)
        relu_4e_double_3x3_1 = F.relu(bn_4e_double_3x3_1)
        ch_concat_4e_chconcat = torch.cat((relu_4e_3x3, relu_4e_double_3x3_1, max_pool_4e_pool), 1)
        conv_5a_1x1     = self.conv_5a_1x1(ch_concat_4e_chconcat)
        conv_5a_3x3_reduce = self.conv_5a_3x3_reduce(ch_concat_4e_chconcat)
        conv_5a_double_3x3_reduce = self.conv_5a_double_3x3_reduce(ch_concat_4e_chconcat)
        avg_pool_5a_pool = F.avg_pool2d(ch_concat_4e_chconcat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False, count_include_pad=False)
        bn_5a_1x1       = self.bn_5a_1x1(conv_5a_1x1)
        bn_5a_3x3_reduce = self.bn_5a_3x3_reduce(conv_5a_3x3_reduce)
        bn_5a_double_3x3_reduce = self.bn_5a_double_3x3_reduce(conv_5a_double_3x3_reduce)
        conv_5a_proj    = self.conv_5a_proj(avg_pool_5a_pool)
        relu_5a_1x1     = F.relu(bn_5a_1x1)
        relu_5a_3x3_reduce = F.relu(bn_5a_3x3_reduce)
        relu_5a_double_3x3_reduce = F.relu(bn_5a_double_3x3_reduce)
        bn_5a_proj      = self.bn_5a_proj(conv_5a_proj)
        conv_5a_3x3_pad = F.pad(relu_5a_3x3_reduce, (1, 1, 1, 1))
        conv_5a_3x3     = self.conv_5a_3x3(conv_5a_3x3_pad)
        conv_5a_double_3x3_0_pad = F.pad(relu_5a_double_3x3_reduce, (1, 1, 1, 1))
        conv_5a_double_3x3_0 = self.conv_5a_double_3x3_0(conv_5a_double_3x3_0_pad)
        relu_5a_proj    = F.relu(bn_5a_proj)
        bn_5a_3x3       = self.bn_5a_3x3(conv_5a_3x3)
        bn_5a_double_3x3_0 = self.bn_5a_double_3x3_0(conv_5a_double_3x3_0)
        relu_5a_3x3     = F.relu(bn_5a_3x3)
        relu_5a_double_3x3_0 = F.relu(bn_5a_double_3x3_0)
        conv_5a_double_3x3_1_pad = F.pad(relu_5a_double_3x3_0, (1, 1, 1, 1))
        conv_5a_double_3x3_1 = self.conv_5a_double_3x3_1(conv_5a_double_3x3_1_pad)
        bn_5a_double_3x3_1 = self.bn_5a_double_3x3_1(conv_5a_double_3x3_1)
        relu_5a_double_3x3_1 = F.relu(bn_5a_double_3x3_1)
        ch_concat_5a_chconcat = torch.cat((relu_5a_1x1, relu_5a_3x3, relu_5a_double_3x3_1, relu_5a_proj), 1)
        conv_5b_1x1     = self.conv_5b_1x1(ch_concat_5a_chconcat)
        conv_5b_3x3_reduce = self.conv_5b_3x3_reduce(ch_concat_5a_chconcat)
        conv_5b_double_3x3_reduce = self.conv_5b_double_3x3_reduce(ch_concat_5a_chconcat)
        max_pool_5b_pool_pad = F.pad(ch_concat_5a_chconcat, (1, 1, 1, 1), value=float('-inf'))
        max_pool_5b_pool = F.max_pool2d(max_pool_5b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        bn_5b_1x1       = self.bn_5b_1x1(conv_5b_1x1)
        bn_5b_3x3_reduce = self.bn_5b_3x3_reduce(conv_5b_3x3_reduce)
        bn_5b_double_3x3_reduce = self.bn_5b_double_3x3_reduce(conv_5b_double_3x3_reduce)
        conv_5b_proj    = self.conv_5b_proj(max_pool_5b_pool)
        relu_5b_1x1     = F.relu(bn_5b_1x1)
        relu_5b_3x3_reduce = F.relu(bn_5b_3x3_reduce)
        relu_5b_double_3x3_reduce = F.relu(bn_5b_double_3x3_reduce)
        bn_5b_proj      = self.bn_5b_proj(conv_5b_proj)
        conv_5b_3x3_pad = F.pad(relu_5b_3x3_reduce, (1, 1, 1, 1))
        conv_5b_3x3     = self.conv_5b_3x3(conv_5b_3x3_pad)
        conv_5b_double_3x3_0_pad = F.pad(relu_5b_double_3x3_reduce, (1, 1, 1, 1))
        conv_5b_double_3x3_0 = self.conv_5b_double_3x3_0(conv_5b_double_3x3_0_pad)
        relu_5b_proj    = F.relu(bn_5b_proj)
        bn_5b_3x3       = self.bn_5b_3x3(conv_5b_3x3)
        bn_5b_double_3x3_0 = self.bn_5b_double_3x3_0(conv_5b_double_3x3_0)
        relu_5b_3x3     = F.relu(bn_5b_3x3)
        relu_5b_double_3x3_0 = F.relu(bn_5b_double_3x3_0)
        conv_5b_double_3x3_1_pad = F.pad(relu_5b_double_3x3_0, (1, 1, 1, 1))
        conv_5b_double_3x3_1 = self.conv_5b_double_3x3_1(conv_5b_double_3x3_1_pad)
        bn_5b_double_3x3_1 = self.bn_5b_double_3x3_1(conv_5b_double_3x3_1)
        relu_5b_double_3x3_1 = F.relu(bn_5b_double_3x3_1)
        ch_concat_5b_chconcat = torch.cat((relu_5b_1x1, relu_5b_3x3, relu_5b_double_3x3_1, relu_5b_proj), 1)
        global_pool     = F.avg_pool2d(ch_concat_5b_chconcat, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        flatten         = global_pool.view(global_pool.size(0), -1)
        fc1             = self.fc1(flatten)
        softmax         = F.softmax(fc1)
        return softmax


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer


synset_mapping = [
6037, 5384, 3694, 7284, 6556, 7721, 1753, 1778, 3288, 1536, 964, 1639, 968, 1100, 2663, 2158, 7124, 3391, 1452, 1401,
7200, 5502, 3259, 1323, 6265, 5713, 7157, 7156, 6732, 7253, 4988, 4116, 6548, 4578, 7707, 7906, 6944, 2176, 9688, 2764,
7491, 8261, 6195, 9777, 8187, 6222, 3693, 7542, 711, 5055, 3261, 6371, 8784, 8030, 2879, 4970, 7076, 2662, 8119, 9539,
9124, 5804, 7441, 7345, 8305, 6399, 7964, 4614, 5098, 1872, 7101, 4890, 7193, 9220, 1690, 3088, 601, 1360, 6851, 2123,
9604, 6779, 3790, 8503, 918, 3254, 8516, 5433, 799, 1633, 828, 6901, 3547, 5482, 1786, 7389, 6381, 3202, 6922, 513,
1915, 6879, 2088, 6723, 591, 2, 4190, 498, 557, 713, 9202, 9211, 1241, 169, 929, 18, 7686, 7963, 4388, 4348,
5182, 7292, 6168, 6586, 2283, 1247, 9439, 5339, 6891, 2720, 3468, 1686, 1278, 3945, 1851, 5176, 3132, 4072, 4234, 2491,
1096, 1330, 5491, 1118, 3623, 59, 383, 1830, 2411, 7301, 2658, 286, 12467, 2808, 598, 1, 7695, 1423, 9996, 3,
6035, 434, 229, 5584, 10847, 11892, 12752, 11701, 8614, 6409, 2967, 2894, 449, 5944, 6395, 12339, 6634, 2112, 326, 41,
2165, 8527, 750, 7930, 2858, 7489, 6113, 0, 10855, 7077, 10127, 4580, 1661, 6710, 8080, 944, 2228, 2779, 7098, 3048,
9003, 2549, 646, 657, 937, 2003, 9334, 572, 157, 28, 8547, 9, 4, 1274, 3232, 957, 7131, 1090, 6656, 573,
6832, 10409, 3253, 895, 2169, 645, 4622, 3402, 8024, 2920, 538, 1084, 1049, 1040, 4092, 299, 473, 128, 4619, 3785,
5061, 11784, 244, 6700, 1319, 11, 654, 720, 761, 499, 2676, 1346, 8865, 158, 3284, 6432, 342, 1806, 466, 561,
3920, 2508, 13998, 1201, 1086, 174, 9423, 86, 14079, 191, 7638, 5661, 425, 3267, 9339, 5601, 639, 151, 8652, 7179,
6769, 818, 1285, 433, 301, 7639, 1099, 1127, 1472, 680, 866, 222, 38, 1263, 250, 1865, 103, 6418, 4623, 8,
5467, 108, 6731, 7414, 2390, 5977, 4346, 2163, 2277, 411, 448, 985, 2478, 1537, 5629, 932, 4064, 1470, 1767, 33,
544, 2214, 441, 6, 419, 1852, 890, 1510, 5108, 5489, 2932, 6324, 6464, 4543, 5825, 5937, 647, 3776, 503, 3349,
1020, 1071, 2750, 1143, 1549, 5070, 3509, 522, 3479, 2356, 924, 3441, 1183, 1600, 3599, 2534, 7455, 8814, 5860, 7498,
746, 6902, 2758, 2876, 6825, 142, 110, 913, 3437, 5196, 9307, 7984, 497, 620, 1042, 1734, 8548, 1080, 5043, 1019,
9459, 1928, 1206, 418, 7459, 460, 13, 390, 176, 7243, 7813, 6141, 10632, 2302, 7609, 7282, 870, 6108, 516, 3373,
7810, 3863, 63, 2265, 4078, 3579, 4996, 880, 3185, 6739, 8002, 6880, 2365, 5377, 1511, 4383, 4352, 3743, 2089, 2648,
309, 592, 162, 6545, 659, 941, 7290, 5401, 2076, 7668, 6142, 4125, 1935, 6669, 2715, 4977, 1739, 182, 7600, 9532,
2348, 2729, 6233, 438, 4404, 669, 5750, 1674, 1985, 7922, 3034, 8978, 6553, 5619, 2117, 5536, 6356, 2677, 417, 5802,
2722, 7883, 708, 1158, 5824, 8822, 4671, 4390, 2206, 7096, 227, 4236, 5981, 5959, 954, 549, 586, 6810, 512, 2432,
279, 6095, 7302, 1643, 2061, 5731, 397, 1293, 5187, 820, 5768, 843, 5790, 1445, 7902, 1538, 3466, 2162, 1544, 7848,
605, 6687, 6417, 5062, 15, 771, 7045, 6419, 16, 5740, 560, 125, 1108, 35, 3140, 1922, 3170, 140, 2823, 6315,
1123, 5249, 8569, 3764, 6753, 4736, 1740, 1171, 7230, 5540, 6521, 1823, 17, 500, 1721, 5746, 8475, 5065, 5083, 3674,
5334, 105, 2966, 566, 2392, 4172, 489, 4313, 722, 7251, 6106, 5976, 803, 366, 1744, 1835, 8121, 3903, 388, 775,
3993, 1064, 1862, 814, 1952, 2433, 350, 5396, 1496, 5683, 3702, 21, 1651, 2028, 5203, 1149, 5379, 8155, 1363, 2308,
2493, 603, 5267, 8806, 8465, 8755, 3118, 1556, 8150, 2834, 8162, 5612, 5106, 906, 187, 1154, 8530, 5707, 7983, 7049,
439, 7594, 5998, 4507, 5296, 2871, 453, 348, 115, 1343, 2126, 5859, 6489, 446, 3193, 5838, 6312, 1750, 196, 553,
1576, 30, 1582, 5413, 2523, 3681, 590, 378, 1596, 3128, 7736, 3943, 847, 1420, 5662, 8323, 1128, 2919, 5889, 9603,
526, 2121, 582, 4248, 8026, 5610, 3942, 6750, 6256, 1275, 36, 748, 1232, 7823, 2728, 826, 723, 4245, 1932, 2248,
5387, 3166, 3287, 8945, 4065, 3233, 3353, 1642, 5884, 2035, 1258, 481, 5641, 2552, 8474, 9711, 8144, 6204, 6206, 2013,
6920, 61, 1000, 2114, 1220, 87, 6526, 4029, 349, 10594, 5118, 5990, 5208, 4374, 5496, 3180, 5222, 4624, 5500, 6565,
5246, 4033, 7014, 3926, 2375, 7574, 10028, 431, 5718, 1779, 3169, 2394, 7824, 3853, 5368, 8867, 628, 1173, 4512, 4945,
5914, 2686, 7730, 8933, 10060, 134, 6350, 8327, 8022, 9961, 6937, 8085, 3682, 4230, 4842, 3146, 101, 3152, 4981, 6604,
1535, 621, 109, 8070, 1065, 1023, 3033, 4036, 1392, 6461, 931, 733, 3862, 5747, 4687, 2174, 415, 1222, 6042, 2031,
1554, 1843, 4802, 4175, 1016, 1107, 7036, 6990, 884, 4355, 749, 3089, 4375, 1028, 1613, 3469, 408, 3960, 2027, 2559,
3859, 2263, 2010, 4220, 1272, 5699, 40, 5663, 4810, 1446, 6004, 1290, 4206, 6778, 112, 4910, 6441, 4919, 4006, 5005,
173, 4841, 2059, 6538, 2476, 1980, 2405, 5908, 5963, 1705, 10068, 7845, 5774, 5362, 5698, 842, 2258, 5513, 531, 3905,
5091, 6963, 5050, 1459, 5001, 2776, 1800, 285, 2917, 2468, 1484, 1863, 859, 6811, 6231, 6001, 1414, 2649, 4335, 2321,
3770, 5349, 6988, 25, 536, 2734, 682, 963, 6790, 622, 0, 6370, 5558, 4204, 5613, 6889, 8533, 7488, 1400, 1791,
9340, 2282, 2363, 3482, 2847, 8346, 320, 2600, 2621, 3487, 4811, 3087, 5244, 2810, 1030, 406, 4956, 4972, 3011, 2014,
4035, 4955, 476, 133, 4134, 8067, 68, 5875, 3980, 2145, 2866, 6060, 4556, 1514, 26, 6494, 811, 2540, 3996, 1669,
5317, 10610, 6268, 613, 8340, 9676, 8371, 3361, 1534, 6203, 4268, 482, 6125, 3739, 6843, 5495, 164, 4872, 4287, 1384,
3751, 9254, 1813, 3823, 4030, 5261, 8853, 3956, 2053, 1599, 7139, 2998, 3247, 4187, 3380, 3639, 7956, 4162, 5414, 6564,
9121, 6638, 9139, 3175, 6010, 1650, 9740, 225, 4649, 1003, 1169, 2732, 2385, 4560, 1390, 1473, 7358, 697, 899, 2803,
4654, 3543, 3272, 2780, 7536, 2736, 2144, 3047, 3510, 8053, 71, 1135, 465, 270, 631, 308, 757, 5103, 7, 1521,
1432, 2284, 1236, 2593, 923, 510, 4031, 2196, 4367, 1887, 7391, 5618, 804, 8727, 5348, 8519, 7911, 800, 6814, 3715]


def classifier_loader():
    return KitModel(load_model_checkpoint_bytes('bninception-imagenet21k'))


registry.add_model(
    Model(
        name = 'bninception-imagenet21k',
        arch = 'bninception',
        transform = StandardTransform(img_resize_size=256, img_crop_size=224),
        normalization = StandardNormalization(mean=[0.4588235294117647, 0.4588235294117647, 0.4588235294117647],
                                              std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098]),
        classifier_loader = classifier_loader,
        eval_batch_size = 40,
        classify = lambda images, model: model(images).t()[synset_mapping].t()
    )
)
