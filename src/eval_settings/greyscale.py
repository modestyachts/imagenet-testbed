from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset
from eval_settings.image_utils import greyscale


def corrupt_greyscale(image):
    return greyscale(image)


registry.add_eval_setting(
    EvalSetting(
        name = 'greyscale',
        dataset = StandardDataset(name='val'),
        size = 50000,
        perturbation_fn_cpu = corrupt_greyscale,
    )
)
