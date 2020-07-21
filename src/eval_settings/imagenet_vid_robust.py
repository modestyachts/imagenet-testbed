import pathlib
import json
import numpy as np

from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset


def validate_vid_robust(logits, targets, image_paths, py_model, merge_op='max'):
    if 'smoothing' in py_model.name:
        merge_op = 'sum'

    with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/class_idx_map.json').resolve()) as f:
        CLASS_IDX_LIST = sorted(list(map(int, json.loads(f.read()).keys())))

    with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/rev_class_idx_map.json').resolve()) as f:
        REV_CLASS_IDX_MAP = json.loads(f.read())

    with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/labels.json').resolve()) as f:
        labels = json.load(f)

    with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/pmsets.json').resolve()) as f:
        pmsets = json.loads(f.read())

    def project_imagenet_predictions_imagenet_vid(preds, merge_op='max'):
        new_preds = np.zeros((preds.shape[0], 30))
        for k,v in REV_CLASS_IDX_MAP.items():
            v = list(map(lambda x: CLASS_IDX_LIST.index(x), v))
            if (merge_op == 'mean'):
                new_preds[:, int(k)] = np.mean(preds[:, v], axis=1).squeeze()
            elif (merge_op == 'median'):
                new_preds[:, int(k)] = np.median(preds[:, v], axis=1).squeeze()
            elif (merge_op == 'max'):
                new_preds[:, int(k)] = np.max(preds[:, v], axis=1).squeeze()
            elif (merge_op == 'sum'):
                new_preds[:, int(k)] = np.sum(preds[:, v], axis=1)
            else:
                raise Exception(f'unsupported merge operation {merge_op} not allowed')
        return new_preds

    def score_predictions(preds, pmsets, labels):
        correct_anchor = 0
        correct_pmk = 0
        N = len(pmsets)
        wrong_map = {}
        for anchor, pmset in pmsets.items():
            pmset_correct = 0
            wrongs = []
            for elem in pmset:
                if np.argmax(preds[elem]) in labels[elem]:
                    pmset_correct += 1
                else:
                    wrongs.append(elem)

            if np.argmax(preds[anchor]) in labels[anchor]:
                correct_anchor  += 1
                pmset_correct += 1
                if len(wrongs) > 0:
                    wrong_map[anchor] = wrongs[-1]

            if pmset_correct == len(pmset) + 1:
                correct_pmk += 1

        return correct_anchor/N, correct_pmk/N

    preds_dict = {}
    logits_projected = project_imagenet_predictions_imagenet_vid(logits.numpy(), merge_op)
    for i, img_name in enumerate(image_paths):
        preds_dict['val/' + img_name] =  logits_projected[i]

    benign,pmk = score_predictions(preds_dict, pmsets, labels)
    metrics_dict = {}
    metrics_dict['pm0'] = benign
    metrics_dict['pm10'] = pmk
    metrics_dict['merge_op'] = merge_op
    return metrics_dict


def validate_val_on_vid_robust_classes(logits, targets, image_paths, py_model, merge_op='max'):
    if 'smoothing' in py_model.name:
        merge_op = 'sum'

    with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/class_idx_map.json').resolve()) as f:
        CLASS_IDX_LIST = sorted(list(map(int, json.loads(f.read()).keys())))

    with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/class_idx_map.json').resolve()) as f:
        CLASS_IDX_MAP = {int(k): v for k, v in json.loads(f.read()).items()}

    with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/rev_class_idx_map.json').resolve()) as f:
        REV_CLASS_IDX_MAP = json.loads(f.read())

    with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/labels.json').resolve()) as f:
        labels = json.load(f)

    def project_imagenet_predictions_imagenet_vid(preds, merge_op='max'):
        new_preds = np.zeros((preds.shape[0], 30))
        for k,v in REV_CLASS_IDX_MAP.items():
            v = list(map(lambda x: CLASS_IDX_LIST.index(x), v))
            if (merge_op == 'mean'):
                new_preds[:, int(k)] = np.mean(preds[:, v], axis=1).squeeze()
            elif (merge_op == 'median'):
                new_preds[:, int(k)] = np.median(preds[:, v], axis=1).squeeze()
            elif (merge_op == 'max'):
                new_preds[:, int(k)] = np.max(preds[:, v], axis=1).squeeze()
            elif (merge_op == 'sum'):
                new_preds[:, int(k)] = np.sum(preds[:, v], axis=1)
            else:
                raise Exception(f'unsupported merge operation {merge_op} not allowed')
        return new_preds

    def score_predictions(logits_projected, targets):
        mask = np.isin(targets, CLASS_IDX_LIST)
        logits_projected, targets = logits_projected[mask], targets[mask]
        targets = [CLASS_IDX_MAP[x] for x in targets]
        preds = logits_projected.argmax(axis=1)

        acc = np.mean(np.equal(preds, targets))

        weights = np.zeros((30,))
        for _, target in labels.items():
            weights[target] += 1
        weights /= weights.sum()

        indv_acc = np.zeros((30,))
        for i in range(30):
            indv_acc[i] = np.mean(np.equal(preds[np.equal(targets, i)], i))

        return {'top1': acc * 100, 
                'top1_uniform_class_weight': indv_acc.mean() * 100,
                'top1_vid_robust_balanced_class_weight': np.sum(indv_acc * weights) * 100}

    logits_projected = project_imagenet_predictions_imagenet_vid(logits.numpy(), merge_op)

    metrics_dict = score_predictions(logits_projected, targets.numpy())
    metrics_dict['merge_op'] = merge_op
    return metrics_dict


with open((pathlib.Path(__file__).parent / 'imagenet-vid-robust_metadata/class_idx_map.json').resolve()) as f:
	class_sublist = sorted(list(map(int, json.loads(f.read()).keys())))

registry.add_eval_setting(
    EvalSetting(
        name = 'imagenet-vid-robust',
        dataset = StandardDataset(name='imagenet-vid-robust'),
        size = 1109,
        metrics_fn = validate_vid_robust,
        class_sublist = class_sublist,
    )
)


idx_subsample_list = [range(x*50, (x+1)*50) for x in class_sublist]
idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

registry.add_eval_setting(
    EvalSetting(
        name = 'val-on-vid-robust-classes',
        dataset = StandardDataset(name='val'),
        size = 14400,
        metrics_fn = validate_val_on_vid_robust_classes,
        class_sublist = class_sublist,
        idx_subsample_list = idx_subsample_list,
    )
)
