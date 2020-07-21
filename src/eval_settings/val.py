import torch

from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset, accuracy_topk
from eval_settings.eval_setting_subsample import class_sublist_1_8


registry.add_eval_setting(
    EvalSetting(
        name = 'val',
        dataset = StandardDataset(name='val'),
        size = 50000,
    )
)

def accuracy_topk_subselected(logits, targets):
    targets = torch.tensor([class_sublist_1_8.index(x) for x in targets])
    return accuracy_topk(logits, targets)

idx_subsample_list = [range(x*50, (x+1)*50) for x in class_sublist_1_8]
idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

registry.add_eval_setting(
    EvalSetting(
        name = 'val_subsampled_class_1_8',
        dataset = StandardDataset(name='val'),
        size = 6250,
        class_sublist = class_sublist_1_8,
        metrics_fn = accuracy_topk_subselected,
        idx_subsample_list = idx_subsample_list,
    )
)
