from inspect import signature

from mldb.utils import download_dataset, DATASET_NAMES


class EvalSetting():
    def __init__(self, name, dataset, size, perturbation_fn_cpu=None, perturbation_fn_gpu=None,
                 metrics_fn=None, adversarial_attack=None, class_sublist=None, idx_subsample_list=None,
                 parent_eval_setting=None, transform=None):
        super().__init__()

        self.name = name
        self.dataset = dataset
        self.size = size
        self.perturbation_fn_cpu = perturbation_fn_cpu
        self.perturbation_fn_gpu = perturbation_fn_gpu
        self.class_sublist = class_sublist
        self.adversarial_attack = adversarial_attack
        self.idx_subsample_list = idx_subsample_list
        self.metrics_fn = metrics_fn if metrics_fn is not None else accuracy_topk
        self.parent_eval_setting = parent_eval_setting
        self.transform = transform

    def get_dataset_root(self):
        return self.dataset.get_root()

    def get_metrics(self, logits, targets, image_paths, py_model):
        sig = signature(self.metrics_fn).parameters.keys()
        assert 'logits' in sig and 'targets' in sig, 'Unrecognized metrics function ' + \
            'definition. Make sure function takes arguments "logits" and "targets"'

        kwargs = {'logits': logits, 'targets': targets}
        if 'image_paths' in sig:
            kwargs['image_paths'] = image_paths
        if 'py_model' in sig:
            kwargs['py_model'] = py_model

        return self.metrics_fn(**kwargs)

    def get_perturbation_fn_gpu(self, py_model):
        if self.adversarial_attack and 'adversarial_attack' in py_model.classify_fn_args:
            return None
        else:
            return self.perturbation_fn_gpu

    def get_perturbation_fn_cpu(self, py_model):
        return self.perturbation_fn_cpu

    def get_idx_subsample_list(self, py_model):
        return self.idx_subsample_list


def accuracy_topk(logits, targets, topk=(1, 5)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res[f'top{k}'] = correct_k.mul_(100.0 / batch_size).item()
    return res


class StandardDataset():
    def __init__(self, name=None, path=None):
        super().__init__()

        assert bool(name) ^ bool(path), \
               'Please specify one (and exactly one) of name or path'

        if name is not None:
            assert name in DATASET_NAMES, \
                f'Dataset {name} is not recognized as an existing dataset in the server.'

        self.name = name
        self.path = path

    def get_root(self):
        if self.name is not None:
            return download_dataset(self.name)
        return self.path
