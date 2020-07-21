import torch

from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset, accuracy_topk
from eval_settings.eval_setting_subsample import class_sublist_1_8


imagenet_v2_datasets = [
'imagenetv2-matched-frequency',
'imagenetv2-matched-frequency-format-val',
'imagenetv2-threshold-0.7-format-val',
'imagenetv2-threshold0.7',
'imagenetv2-top-images-format-val',
'imagenetv2-topimages']


for imagenet_v2_dataset in imagenet_v2_datasets:
    registry.add_eval_setting(
        EvalSetting(
            name = imagenet_v2_dataset,
            dataset = StandardDataset(name=imagenet_v2_dataset),
            size = 10000,
        )
    )


def accuracy_topk_subselected(logits, targets):
    targets = torch.tensor(list(map(lambda x: class_sublist_1_8.index(x), targets)))
    return accuracy_topk(logits, targets)

idx_subsample_list = [range(x*10, (x+1)*10) for x in class_sublist_1_8]
idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

registry.add_eval_setting(
    EvalSetting(
        name = 'imagenetv2-matched-frequency-format-val_subsampled_class_1_8',
        dataset = StandardDataset(name='imagenetv2-matched-frequency-format-val'),
        size = 1250,
        class_sublist = class_sublist_1_8,
        metrics_fn = accuracy_topk_subselected,
        idx_subsample_list = idx_subsample_list,
    )
)
