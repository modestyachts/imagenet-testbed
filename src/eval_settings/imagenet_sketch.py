from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset


registry.add_eval_setting(
    EvalSetting(
        name = 'imagenet-sketch',
        dataset = StandardDataset(name='imagenet-sketch'),
        size = 50000,
    )
)
