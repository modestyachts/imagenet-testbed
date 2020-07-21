import numpy as np
import cleverhans.future.torch.attacks.projected_gradient_descent as pgd

from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset
from eval_settings.eval_setting_subsample import idx_subsample_list_50k_10percent


pgd_attack_params = {   
'pgd.l2.eps0.1': {   'eps': 0.1,
                     'norm': 2,
                     'num_steps': 100,
                     'step_size': 0.01,
                     'targeted': False,
                     'attack_style': 'pgd'},
'pgd.l2.eps0.5': {   'eps': 0.5,
                     'norm': 2,
                     'num_steps': 100,
                     'step_size': 0.05,
                     'targeted': False,
                     'attack_style': 'pgd'},
'pgd.linf.eps0.5': {   'eps': 0.001960784314,
                       'norm': np.inf,
                       'num_steps': 100,
                       'step_size': 5.88235294117647e-05,
                       'targeted': False,
                       'attack_style': 'pgd'},
'pgd.linf.eps2': {   'eps': 0.007843137255,
                     'norm': np.inf,
                     'num_steps': 100,
                     'step_size': 0.0002352941176470588,
                     'targeted': False,
                     'attack_style': 'pgd'},
# 'pgd.linf.eps2.targeted': {  'eps': 2,
#                              'norm': np.inf,
#                              'num_steps': 100,
#                              'step_size': 0.0002352941176470588,
#                              'targeted': True,
#                              'attack_style': 'pgd'}
}


def pgd_style_attack(d, images, model):
    eps = d['eps']
    if d['norm'] == 2:
        # attack calibrated for size 224, so modify l2 norm for specific image size
        eps = eps * images.size(2) / 224
    return pgd(model, images, eps, d['step_size'], d['num_steps'], d['norm'], targeted=d['targeted'])

def gen_attack_fn(d):
    return lambda images, model: pgd_style_attack(d, images, model)


for pgd_attack_name, d in pgd_attack_params.items():
    registry.add_eval_setting(
        EvalSetting(
            name = pgd_attack_name,
            dataset = StandardDataset(name='val'),
            size = 50000,
            perturbation_fn_gpu = gen_attack_fn(d),
            adversarial_attack = d,
        )
    )
    registry.add_eval_setting(
        EvalSetting(
            name = pgd_attack_name+'_10percent',
            dataset = StandardDataset(name='val'),
            size = 5000,
            perturbation_fn_gpu = gen_attack_fn(d),
            adversarial_attack = d,
            idx_subsample_list = idx_subsample_list_50k_10percent,
            parent_eval_setting = pgd_attack_name,
        )
    )
