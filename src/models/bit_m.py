from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


# THE FOLLOWING SECTION OF THE CODE FOLLOWS THIS LICENSE:

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code is adapted from https://github.com/google-research/big_transfer

class StdConv2d(nn.Conv2d):

  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                   padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias)


def tf2th(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
  """Pre-activation (v2) bottleneck block.
  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, cin, cout=None, cmid=None, stride=1):
    super().__init__()
    cout = cout or cin
    cmid = cmid or cout//4

    self.gn1 = nn.GroupNorm(32, cin)
    self.conv1 = conv1x1(cin, cmid)
    self.gn2 = nn.GroupNorm(32, cmid)
    self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
    self.gn3 = nn.GroupNorm(32, cmid)
    self.conv3 = conv1x1(cmid, cout)
    self.relu = nn.ReLU(inplace=True)

    if (stride != 1 or cin != cout):
      # Projection also with pre-activation according to paper.
      self.downsample = conv1x1(cin, cout, stride)

  def forward(self, x):
    out = self.relu(self.gn1(x))

    # Residual branch
    residual = x
    if hasattr(self, 'downsample'):
      residual = self.downsample(out)

    # Unit's branch
    out = self.conv1(out)
    out = self.conv2(self.relu(self.gn2(out)))
    out = self.conv3(self.relu(self.gn3(out)))

    return out + residual

  def load_from(self, weights, prefix=''):
    convname = 'standardized_conv2d'
    with torch.no_grad():
      self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
      self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
      self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
      self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
      self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
      self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
      self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
      self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
      self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
      if hasattr(self, 'downsample'):
        w = weights[f'{prefix}a/proj/{convname}/kernel']
        self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
  """Implementation of Pre-activation (v2) ResNet mode."""

  def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
    super().__init__()
    wf = width_factor  # shortcut 'cause we'll use it a lot.

    # The following will be unreadable if we split lines.
    # pylint: disable=line-too-long
    self.root = nn.Sequential(OrderedDict([
        ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
        ('pad', nn.ConstantPad2d(1, 0)),
        ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        # The following is subtly not the same!
        # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    self.body = nn.Sequential(OrderedDict([
        ('block1', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf)) for i in range(2, block_units[0] + 1)],
        ))),
        ('block2', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
        ))),
        ('block3', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
        ))),
        ('block4', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
        ))),
    ]))
    # pylint: enable=line-too-long

    self.zero_head = zero_head
    self.head = nn.Sequential(OrderedDict([
        ('gn', nn.GroupNorm(32, 2048*wf)),
        ('relu', nn.ReLU(inplace=True)),
        ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
        ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
    ]))

  def forward(self, x):
    x = self.head(self.body(self.root(x)))
    assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
    return x[...,0,0]

  def load_from(self, weights, prefix='resnet/'):
    with torch.no_grad():
      self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
      self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
      self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
      if self.zero_head:
        nn.init.zeros_(self.head.conv.weight)
        nn.init.zeros_(self.head.conv.bias)
      else:
        self.head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
        self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

      for bname, block in self.body.named_children():
        for uname, unit in block.named_children():
          unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])

# END LICENSED SECTION


classes_21k_to_1k_mapping = [
359,368,460,475,486,492,496,514,516,525,547,548,556,563,575,641,648,723,733,765,
801,826,852,858,878,896,900,905,908,910,935,946,947,994,999,1003,1005,1010,1027,1029,
1048,1055,1064,1065,1069,1075,1079,1081,1085,1088,1093,1106,1143,1144,1145,1147,1168,1171,1178,1187,
1190,1197,1205,1216,1223,1230,1236,1241,1245,1257,1259,1260,1267,1268,1269,1271,1272,1273,1277,1303,
1344,1349,1355,1357,1384,1388,1391,1427,1429,1432,1437,1450,1461,1462,1474,1502,1503,1512,1552,1555,
1577,1584,1587,1589,1599,1615,1616,1681,1692,1701,1716,1729,1757,1759,1764,1777,1786,1822,1841,1842,
1848,1850,1856,1860,1861,1864,1876,1897,1898,1910,1913,1918,1922,1928,1932,1935,1947,1951,1953,1970,
1977,1979,2001,2017,2067,2081,2087,2112,2128,2135,2147,2174,2175,2176,2177,2178,2181,2183,2184,2187,
2189,2190,2191,2192,2193,2197,2202,2203,2206,2208,2209,2211,2212,2213,2214,2215,2216,2217,2219,2222,
2223,2224,2225,2226,2227,2228,2229,2230,2236,2238,2240,2241,2242,2243,2244,2245,2247,2248,2249,2250,
2251,2252,2255,2256,2257,2262,2263,2264,2265,2266,2268,2270,2271,2272,2273,2275,2276,2279,2280,2281,
2282,2285,2289,2292,2295,2296,2297,2298,2299,2300,2301,2302,2303,2304,2305,2306,2309,2310,2312,2313,
2314,2315,2316,2318,2319,2321,2322,2326,2329,2330,2331,2332,2334,2335,2336,2337,2338,2339,2341,2342,
2343,2344,2346,2348,2349,2351,2352,2353,2355,2357,2358,2359,2360,2364,2365,2368,2369,2377,2382,2383,
2385,2397,2398,2400,2402,2405,2412,2421,2428,2431,2432,2433,2436,2441,2445,2450,2453,2454,2465,2469,
2532,2533,2538,2544,2547,2557,2565,2578,2612,2658,2702,2722,2731,2738,2741,2747,2810,2818,2833,2844,
2845,2867,2874,2882,2884,2888,2889,3008,3012,3019,3029,3033,3042,3091,3106,3138,3159,3164,3169,3280,
3296,3311,3318,3320,3324,3330,3366,3375,3381,3406,3419,3432,3434,3435,3493,3495,3503,3509,3511,3513,
3517,3521,3526,3546,3554,3600,3601,3606,3612,3613,3616,3622,3623,3627,3632,3634,3636,3638,3644,3646,
3649,3650,3651,3656,3663,3673,3674,3689,3690,3702,3733,3769,3971,3974,4065,4068,4073,4102,4136,4140,
4151,4159,4165,4207,4219,4226,4249,4256,4263,4270,4313,4321,4378,4386,4478,4508,4512,4536,4542,4550,
4560,4562,4570,4571,4572,4583,4588,4594,4604,4608,4623,4634,4636,4646,4651,4652,4686,4688,4691,4699,
4724,4727,4737,4770,4774,4789,4802,4807,4819,4880,4886,4908,4927,4931,4936,4964,4976,4993,5028,5033,
5043,5046,5096,5111,5114,5131,5132,5183,5199,5235,5275,5291,5293,5294,5343,5360,5362,5364,5390,5402,
5418,5428,5430,5437,5443,5473,5484,5486,5505,5507,5508,5510,5567,5578,5580,5584,5606,5613,5629,5672,
5676,5692,5701,5760,5769,5770,5779,5814,5850,5871,5893,5911,5949,5954,6005,6006,6012,6017,6023,6024,
6040,6050,6054,6087,6105,6157,6235,6237,6256,6259,6286,6291,6306,6339,6341,6343,6379,6383,6393,6405,
6479,6511,6517,6541,6561,6608,6611,6615,6678,6682,6707,6752,6798,6850,6880,6885,6890,6920,6981,7000,
7009,7038,7049,7050,7052,7073,7078,7098,7111,7165,7198,7204,7280,7283,7286,7287,7293,7294,7305,7318,
7341,7346,7354,7382,7427,7428,7435,7445,7450,7455,7467,7469,7497,7502,7506,7514,7523,7651,7661,7664,
7672,7679,7685,7696,7730,7871,7873,7895,7914,7915,7920,7934,7935,7949,8009,8036,8051,8065,8074,8090,
8112,8140,8164,8168,8178,8182,8198,8212,8216,8230,8242,8288,8289,8295,8318,8352,8368,8371,8375,8376,
8401,8416,8419,8436,8460,8477,8478,8482,8498,8500,8539,8543,8552,8555,8580,8584,8586,8594,8598,8601,
8606,8610,8611,8622,8627,8639,8649,8650,8653,8654,8667,8672,8673,8674,8676,8684,8720,8723,8750,8753,
8801,8815,8831,8835,8842,8845,8858,8897,8916,8951,8954,8959,8970,8976,8981,8983,8989,8991,8993,9019,
9039,9042,9043,9056,9057,9070,9087,9098,9106,9130,9131,9155,9171,9183,9198,9199,9201,9204,9212,9221,
9225,9229,9250,9260,9271,9279,9295,9300,9310,9322,9345,9352,9376,9377,9382,9392,9401,9405,9441,9449,
9464,9475,9502,9505,9514,9515,9545,9567,9576,9608,9609,9624,9633,9639,9643,9656,9674,9740,9752,9760,
9767,9778,9802,9820,9839,9879,9924,9956,9961,9963,9970,9997,10010,10031,10040,10052,10073,10075,10078,10094,
10097,10109,10118,10121,10124,10158,10226,10276,10304,10307,10314,10315,10332,10337,10338,10413,10423,10451,10463,10465,
10487,10519,10522,10523,10532,10534,10535,10551,10559,10574,10583,10586,10589,10612,10626,10635,10638,10677,10683,10726,
10776,10782,10783,10807,10837,10840,10848,10859,10871,10881,10884,10908,10914,10921,10936,10947,10951,10952,10957,10999,
11003,11018,11023,11025,11027,11045,11055,11095,11110,11137,0,11168,11186,11221,11223,11242,11255,11259,11279,11306,
11311,11331,11367,11377,11389,11392,11401,11407,11437,11449,11466,11469,11473,11478,11483,11484,11507,11536,11558,11566,
11575,11584,11594,11611,11612,11619,11621,11640,11643,11664,11674,11689,11709,11710,11716,11721,11726,11729,11743,11760,
11771,11837,11839,11856,11876,11878,11884,11889,11896,11917,11923,11930,11944,11952,11980,11984,12214,12229,12239,12241,
12242,12247,12283,12349,12369,12373,12422,12560,12566,12575,12688,12755,12768,12778,12780,12812,12832,12835,12836,12843,
12847,12849,12850,12856,12858,12873,12938,12971,13017,13038,13046,13059,13085,13086,13088,13094,13134,13182,13230,13406,
13444,13614,13690,13698,13709,13749,13804,13982,14051,14059,14219,14246,14256,14264,14294,14324,14367,14389,14394,14438,
14442,14965,15732,16744,18037,18205,18535,18792,19102,20019,20462,21026,21045,21163,21171,21181,21196,21200,21369,21817,
]


model_params_finetuned = {
'BiT-M-R50x1-ILSVRC2012': {   'arch': 'BiT-M-R50x1',
                              'eval_batch_size': 64,
                              'adversarial_batch_size': 4},
'BiT-M-R50x3-ILSVRC2012': {   'arch': 'BiT-M-R50x3',
                              'eval_batch_size': 32,
                              'adversarial_batch_size': 1},
'BiT-M-R101x1-ILSVRC2012': {  'arch': 'BiT-M-R101x1',
                              'eval_batch_size': 64,
                              'adversarial_batch_size': 1},
'BiT-M-R101x3-ILSVRC2012': {  'arch': 'BiT-M-R101x3',
                              'eval_batch_size': 32,
                              'adversarial_batch_size': 1},
'BiT-M-R152x4-ILSVRC2012': {  'arch': 'BiT-M-R152x4',
                              'eval_batch_size': 8,
                              'adversarial_batch_size': 1}}

model_params_nonfinetuned = {
'BiT-M-R50x1-nonfinetuned': {   'arch': 'BiT-M-R50x1',
                                'head_size': 21843,
                                'eval_batch_size': 64,
                                'adversarial_batch_size': 4},
'BiT-M-R50x3-nonfinetuned': {   'arch': 'BiT-M-R50x3',
                                'head_size': 21843,
                                'eval_batch_size': 32,
                                'adversarial_batch_size': 1},
'BiT-M-R101x1-nonfinetuned': {  'arch': 'BiT-M-R101x1',
                                'head_size': 21843,
                                'eval_batch_size': 64,
                                'adversarial_batch_size': 1},
'BiT-M-R101x3-nonfinetuned': {  'arch': 'BiT-M-R101x3',
                                'head_size': 21843,
                                'eval_batch_size': 32,
                                'adversarial_batch_size': 1},
'BiT-M-R152x4-nonfinetuned': {  'arch': 'BiT-M-R152x4',
                                'head_size': 21843,
                                'eval_batch_size': 8,
                                'adversarial_batch_size': 1}}


def gen_classifier_loader(name, d):
    def classifier_loader():
        model = KNOWN_MODELS[d['arch']](head_size=1000 if 'head_size' not in d else d['head_size'])
        load_model_state_dict(model, name)
        return model
    return classifier_loader

for name, d in model_params_finetuned.items():
    registry.add_model(
        Model(
            name = name,
            arch = d['arch'],
            transform = transforms.Compose([transforms.Resize((480, 480)),
                                            transforms.ToTensor()]),
            normalization = StandardNormalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            classifier_loader = gen_classifier_loader(name, d),
            eval_batch_size = d['eval_batch_size'],
            adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None,
        )
    )

for name, d in model_params_nonfinetuned.items():
    registry.add_model(
        Model(
            name = name,
            arch = d['arch'],
            transform = transforms.Compose([transforms.Resize((480, 480)),
                                            transforms.ToTensor()]),
            normalization = StandardNormalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            classifier_loader = gen_classifier_loader(name, d),
            eval_batch_size = d['eval_batch_size'],
            adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None,
            classify = lambda images, model: model(images).t()[classes_21k_to_1k_mapping].t(),
        )
    )
