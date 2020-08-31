import torch
import numpy as np

from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict


"""
MAKE SURE YOU ARE SPECIFYING ONE AND ONLY ONE GPU TO EVAL.PY WHEN RUNNING THIS MODEL
Also make sure to properly specify store_filename_prefix and load_filename_prefix below
And also only run one evaluation setting each time otherwise the previous ones will be overwritten

A sample workflow looks like:
1. set store_filename_prefix (ie. use the name of the eval setting)
2. python eval.py --gpus 0 --models google_resnet101_jft-300M --eval-settings EVALSETTING
3. load exported images with:
    open(f'prefix_0.npy', 'rb') as f:
        images = np.load(f)
4. save the logits as numpy arrays (make sure to keep the number at the end of the filename)
5. set load_logits_jft to however you decided to name the logits
6. switch classify = store_logits_jft to classify = load_logits_jft (line 63)
6. python eval.py --gpus 0 --models google_resnet101_jft-300M --eval-settings EVALSETTING --db
7. Repeat for the next eval setting
"""

# use this to initially store logits to disk - DO NOT pass the --db flag to eval.py at this stage
store_filename_prefix = ''
store_counter = 0
def store_logits_jft(images, model):
    global store_counter
    assert store_filename_prefix is not ''

    images = images.cpu().permute([0, 2, 3, 1]).numpy()
    with open(f'{store_filename_prefix}_{store_counter}.npy', 'wb') as f:
        np.save(f, images)

    store_counter += 1
    return torch.empty(images.shape[0], 1000).cuda()

# use this to load logits from disk - PASS the --db flag to eval.py at this stage
load_filename_prefix = ''
load_counter = 0
def load_logits_jft(images, model):
    global load_counter
    assert load_filename_prefix is not ''

    with open(f'{load_filename_prefix}_{load_counter}.npy', 'rb') as f:
        logits = torch.from_numpy(np.load(f)).cuda()

    load_counter += 1
    return logits


# registry.add_model(
#     Model(
#         name = 'google_resnet101_jft-300M',
#         arch = 'resnet101',
#         transform = StandardTransform(img_resize_size=256, img_crop_size=224),
#         normalization = StandardNormalization(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         classifier_loader = lambda: None,
#         eval_batch_size = 1000,
#         classify = store_logits_jft, # switch to load_logits_jft when needed
#     )
# )
