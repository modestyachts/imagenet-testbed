from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset


registry.add_eval_setting(
    EvalSetting(
        name = 'stylized_imagenet',
        dataset = StandardDataset(name='stylized_imagenet'),
        size = 50000,
    )
)
