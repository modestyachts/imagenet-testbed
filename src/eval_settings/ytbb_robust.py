
import collections
import csv
import json
import logging
from pathlib import Path
from pprint import pformat
from collections import defaultdict
import pathlib
import json
import numpy as np
import random

from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset


def load_labels(input_paths):
    if isinstance(input_paths, (Path, str)):
        input_paths = [input_paths]

    # Map data to list of (input_path, label) tuples.
    labels = collections.defaultdict(list)
    labels_list = None
    for path in input_paths:
        with open(path, 'r') as f:
            annotations = json.load(f)
            for i, row in enumerate(annotations['annotations']):
                labels[row['key']].append((path, row))
            if labels_list is None:
                labels_list = annotations['labels']
            else:
                assert labels_list == annotations['labels']

    for key, key_labels in labels.items():
        if len(key_labels) > 1:
            paths = [x[0] for x in key_labels]
            logging.debug(
                f'{key} labeled multiple times in {paths}; using latest '
                f'label from {paths[-1]}.')
        labels[key] = key_labels[-1][1]
    return labels, labels_list


def filter_labels(labels,
                  labels_list,
                  file_logger=None,
                  must_have=[],
                  must_not_have=[],
                  can_have=[],
                  must_have_one_of=False,
                  unspecified_labels_policy='error',
                  return_nonmatching=False):
    if file_logger is None:
        file_logger = logging.getLogger()

    label_map = {}
    label_names = {}
    for i, label in enumerate(labels_list):
        label_map[label] = i
        label_names[i] = label

    def validate_label(label):
        if label not in label_map:
            raise ValueError('Unknown label %s, valid labels: %s' %
                             (label, label_map.keys()))
        return True

    must_have_labels = set(
        [label_map[x] for x in must_have if validate_label(x)])
    must_not_have_labels = set(
        [label_map[x] for x in must_not_have if validate_label(x)])
    can_have_labels = set(
        [label_map[x] for x in can_have if validate_label(x)])

    unspecified_labels = (
        set(label_map.values()) -
        (must_have_labels | must_not_have_labels | can_have_labels))
    if unspecified_labels:
        if unspecified_labels_policy == 'error':
            raise ValueError('Label(s): %s were not specified in any of '
                             '--{must,must-not,can}-have.' %
                             [label_names[x] for x in unspecified_labels])
        elif unspecified_labels_policy == 'can-have':
            can_have_labels |= unspecified_labels
        elif unspecified_labels_policy == 'must-have':
            must_have_labels |= unspecified_labels
        elif unspecified_labels_policy == 'must-not-have':
            must_not_have_labels |= unspecified_labels
        else:
            raise ValueError('Unknown unspecified_labels_policy %s' %
                             unspecified_labels_policy)

    logging.info('Looking for rows that')
    if must_have_one_of:
        logging.info('MUST HAVE (one of): %s',
                     [label_names[x] for x in must_have_labels])
    else:
        logging.info('MUST HAVE: %s',
                     [label_names[x] for x in must_have_labels])
    logging.info('MUST NOT HAVE: %s',
                 [label_names[x] for x in must_not_have_labels])
    logging.info('CAN HAVE: %s', [label_names[x] for x in can_have_labels])

    valid_rows = []
    invalid_rows = []
    for key, row in labels.items():
        row_labels = set(row['labels'])
        missing_labels = must_have_labels - row_labels
        unwanted_labels = row_labels & must_not_have_labels
        if must_have_one_of:
            if missing_labels == must_have_labels:
                file_logger.info('Label %s missing labels %s' % (pformat(
                    dict(row)), [label_names[x] for x in missing_labels]))
                invalid_rows.append(row)
                continue
        elif missing_labels:
            file_logger.info(
                'Label %s missing labels %s' %
                (pformat(dict(row)), [label_names[x] for x in missing_labels]))
            invalid_rows.append(row)
            continue
        if unwanted_labels:
            file_logger.info(
                'Label %s has unwanted labels %s' %
                (pformat(dict(row)), [label_names[x]
                                      for x in unwanted_labels]))
            invalid_rows.append(row)
            continue
        valid_rows.append(row)
    if return_nonmatching:
        return valid_rows, invalid_rows
    else:
        return valid_rows



def evaluate_pmk(predictions, labels, valid_pmk):
    """
    Args:
        predictions (Dict[str, np.array])
        labels (Dict[str, List[int]]): Labels for anchor frames.
        valid_pmk (Dict[str, Dict[int, str]])
    """
    anchor_is_correct = {}
    pmk_is_correct = {}
    for anchor, pmk_dict in valid_pmk.items():
        anchor_labels = labels[anchor]
        anchor_prediction = predictions[anchor].argmax()
        anchor_is_correct[anchor] = anchor_prediction in anchor_labels

        pmk_is_correct[anchor] = {}
        for offset, pmk_key in pmk_dict.items():
            pmk_prediction = predictions[pmk_key].argmax()
            pmk_is_correct[anchor][pmk_key] = pmk_prediction in anchor_labels
    return anchor_is_correct, pmk_is_correct


def create_pmk_score(predictions_by_key, anchor_labels, pmk_frames):
    """
    Args:
        predictions_by_key (Dict[str, np.array])
        anchor_labels (Dict[str, List[int]]): Labels for anchor frames.
        pmk_frames (Dict[str, Dict[int, str]]): Map anchor frame to dict
            mapping valid pmk offset to pmk frame key.
    """
    pmk_frames = pmk_frames.copy()
    for anchor in anchor_labels:
        if anchor not in pmk_frames:
            pmk_frames[anchor] = {}

    anchor_is_correct, pmk_is_correct = evaluate_pmk(predictions_by_key,
                                                     anchor_labels, pmk_frames)

    correct_anchors = {
        k for k, correct in anchor_is_correct.items() if correct
    }
    all_anchors = [k for k, correct in anchor_is_correct.items()]
    num_anchor_correct = len(correct_anchors)
    anchor_accuracy = num_anchor_correct / max(len(anchor_is_correct), 1e-9)

    pmk_correct = [
        anchor for anchor in correct_anchors
        if all(pmk_is_correct[anchor].values())
    ]

    rand_correct = [
        anchor for anchor in all_anchors
        if (len(pmk_is_correct[anchor].values()) == 0) or  random.choice(list(pmk_is_correct[anchor].values()))
    ]
    pmk_accuracy = len(pmk_correct) / max(len(anchor_is_correct), 1e-9)
    rand_accuracy = len(rand_correct) / max(len(anchor_is_correct), 1e-9)

    # Collect auxiliary data.
    benign_frames = sorted(anchor_labels.keys())
    adversarial_pmk = {}  # Map anchor to list of adversarial pmk offsets
    nonadversarial_pmk = {}  # Map anchor to list of non-adv pmk offsets

    for anchor in benign_frames:
        if not anchor_is_correct[anchor]:
            adversarial_pmk[anchor] = None
            nonadversarial_pmk[anchor] = None
        else:
            incorrect_frames = []
            correct_frames = []
            for i,(offset, pmk_key) in enumerate(pmk_frames[anchor].items()):
                if pmk_is_correct[anchor][pmk_key]:
                    correct_frames.append(offset)
                else:
                    incorrect_frames.append(offset)
            adversarial_pmk[anchor] = incorrect_frames
            nonadversarial_pmk[anchor] = correct_frames

    score_info = {}
    score_info["benign_accuracy"] = anchor_accuracy
    score_info["benign_frames"] = benign_frames
    score_info["adversarial_pmk"] = adversarial_pmk
    score_info["nonadversarial_pmk"] = nonadversarial_pmk
    score_info["pmk_keys"] = pmk_frames
    score_info["correct_anchors"] = sorted(correct_anchors)
    score_info["incorrect_anchors"] = sorted(
        set(anchor_is_correct.keys()) - correct_anchors)

    score_info["l_infs"] = []  # TODO
    return pmk_accuracy, score_info


def ms_to_frame_15fps(ms):
    return round(ms / 1000 * 15)


def path_to_key(path):
    path = pathlib.Path(path)
    return f"{path.parent.name}/{path.name}"


def get_pmk_key(anchor_key, pmk_index):
    """Returns pmk portion of pmk key.

    The full pm-k key, as used in annotations, is '{anchor_key},{pmk_key}'."""
    video, anchor_index, anchor_ms = parse_frame_key(anchor_key)
    prefix = f'{video}_{anchor_ms}'
    return f'{prefix}/frame-{pmk_index}.jpg'


def split_pmk_key(key):
    anchor_path, pmk_path = key.split(",")
    return path_to_key(anchor_path), path_to_key(pmk_path)


def parse_frame_key(key, return_ms=True):
    """Parse key into video, frame index, and anchor ms."""
    key = path_to_key(key)
    # Key format: <video>_<anchor_timestamp>/frame-<idx>.jpg
    parent, name = key.split('/')
    video, anchor_ms = parent.rsplit('_', 1)
    frame_idx = name.split('-')[1].split('.')[0]
    if return_ms:
        return video, int(frame_idx), int(anchor_ms)
    else:
        return video, int(frame_idx)


class YtbbPmkDataset:
    def __init__(self, anchor_annotations, pmk_annotations):
        # Split anchors into good and bad labels.
        anchors_good, anchors_bad, anchor_label_list = load_anchor_annotations(
            anchor_annotations)

        # Collect final anchor labels.
        self.anchor_valid_annotations = anchors_good
        self.anchor_invalid_annotations = anchors_bad
        self.anchor_label_list = anchor_label_list

        valid_pmk, invalid_pmk, bad_anchors, pmk_label_list = (
            load_pmk_annotations(pmk_annotations))

        self.anchor_invalid_annotations += [
            x for x in self.anchor_valid_annotations if x['key'] in bad_anchors
        ]
        self.anchor_valid_annotations = [
            x for x in self.anchor_valid_annotations
            if x['key'] not in bad_anchors
        ]

        self.pmk_valid_keys = defaultdict(set)
        for annotation in valid_pmk:
            anchor_key, pmk_key = split_pmk_key(annotation['key'])
            self.pmk_valid_keys[anchor_key].add(annotation['key'])

        self.pmk_invalid_keys = defaultdict(set)
        for annotation in invalid_pmk:
            anchor_key, pmk_key = split_pmk_key(annotation['key'])
            self.pmk_invalid_keys[anchor_key].add(annotation['key'])

        self.pmk_label_list = pmk_label_list

    def get_anchor_labels(self,
                          reverse_ytbb_class_index,
                          reviewed=True):
        """
        Args:
            initial_labels (Dict[str, Set[int]]): Initial YTBB labels. We
                require these because the loaded annotations are only
                guaranteed to contain labels that were _added_ to each
                anchor.

        Returns:
            anchor_labels: Map anchor_key to list of YTBB labels.
        """
        initial_labels_csv = (pathlib.Path(__file__).parent / 'ytbb-robust_metadata/ytbb_robustness_test_anchors_full.csv').resolve()

        with open(initial_labels_csv, 'r') as f:
            reader = csv.DictReader(f)
            initial_labels = {}
            for row in reader:
                label = int(row["label"])
                if label == 23:
                    label = 22
                elif label == 22:
                    label = 23
                initial_labels[f'{row["ytid"]}_{row["time_ms"]}'] = {label}

        annotations = self.anchor_valid_annotations
        if not reviewed:
            annotations = annotations + self.anchor_invalid_annotations
        anchor_labels = annotations_to_ytbb_labels(annotations,
                                                   self.anchor_label_list,
                                                   reverse_ytbb_class_index)
        for key, labels in anchor_labels.items():
            anchor_label_key = key.split('/')[0]
            assert isinstance(initial_labels[anchor_label_key], set)
            labels.update(initial_labels[anchor_label_key])
        return anchor_labels

    def get_pmk(self, k=10, reviewed=True):
        """
        Returns:
            pmk_sets (Dict[str, Dict[int, str]]): Map anchor keys to dict
                mapping offset to pmk frame key. If reviewed is True, then
                the mapping will only contain valid (i.e., similar) pmk frames.
        """
        pmk_sets = {}
        for anchor in self.anchor_valid_annotations:
            anchor_key = anchor['key']
            video, anchor_index, anchor_ms = parse_frame_key(
                anchor_key)
            pmk_sets[anchor_key] = {}
            for i in range(-k, k+1):
                if i == 0:
                    continue
                pmk_index = anchor_index + i
                pmk_key = get_pmk_key(anchor_key, pmk_index)
                full_pmk_key = f'{anchor_key},{pmk_key}'
                if (reviewed and
                        full_pmk_key not in self.pmk_valid_keys[anchor_key]):
                    continue
                pmk_sets[anchor_key][i] = pmk_key
        return pmk_sets


def load_anchor_annotations(anchor_paths):
    labels, labels_list = load_labels(anchor_paths)

    for annotation in labels.values():
        annotation['key'] = path_to_key(annotation['key'])

    good_anchors, other_anchors = filter_labels(
        labels,
        labels_list,
        must_have=['good'],
        must_not_have=['bad', 'unsure'],
        unspecified_labels_policy='can-have',
        return_nonmatching=True)
    return good_anchors, other_anchors, labels_list


def annotations_to_ytbb_labels(annotations, label_list, reverse_ytbb_map):
    """Convert annotations from labeling UI to match YTBB index labels.

    Args:
        annotations (List[Dict]): Contains list of annotation objects with keys
            'key', 'notes', 'labels'.
        label_list (List[str]): List of label names.
        reverse_ytbb_map (Dict): Map YTBB label names to indices.

    Returns:
        labels (Dict[str, Set[int]]): Map keys to list of YTBB label indices.
    """
    labels = {}
    for ann in annotations:
        labels_str = [label_list[l] for l in ann['labels']]
        labels[ann['key']] = {
            reverse_ytbb_map[l]
            for l in labels_str if l in reverse_ytbb_map
        }
    return labels


def load_pmk_annotations(pmk_paths):
    problematic_labels = [
        'dissimilar', 'problematic', 'incorrect', 'dont know', 'deformation',
        'background-change', 'occlusion-increased', 'blur-increase',
        'bad-anchor'
    ]

    labels, labels_list = load_labels(pmk_paths)
    # Pairs that do not have a problematic label and are marked similar.
    valid_pmk, bad_pmk = filter_labels(labels,
                                       labels_list,
                                       must_not_have=problematic_labels,
                                       can_have=['correct'],
                                       must_have=['similar'],
                                       unspecified_labels_policy='can-have',
                                       return_nonmatching=True)
    # Choose bad anchors
    bad_anchors = filter_labels(labels,
                                labels_list,
                                must_have=['bad-anchor'],
                                unspecified_labels_policy='can-have')
    bad_anchors = {split_pmk_key(x['key'])[0] for x in bad_anchors}
    valid_pmk = [
        x for x in valid_pmk
        if split_pmk_key(x['key'])[0] not in bad_anchors
    ]

    return valid_pmk, bad_pmk, bad_anchors, labels_list


def validate_ytbb_robust(logits, targets, image_paths, py_model, merge_op='max'):
    PMK_DIST = 10
    if 'smoothing' in py_model.name:
        merge_op = 'sum'

    with open((pathlib.Path(__file__).parent / 'ytbb-robust_metadata/class_idx_map.json').resolve()) as f:
        CLASS_IDX_LIST = sorted(list(map(int, json.loads(f.read()).keys())))

    with open((pathlib.Path(__file__).parent / 'ytbb-robust_metadata/rev_class_idx_map.json').resolve()) as f:
        REV_CLASS_IDX_MAP = json.loads(f.read())

    anchor_annots = (pathlib.Path(__file__).parent / 'ytbb-robust_metadata/anchor_labels.json').resolve()
    pmk_annots = (pathlib.Path(__file__).parent / 'ytbb-robust_metadata/pmk_labels.json').resolve()

    with open((pathlib.Path(__file__).parent / 'ytbb-robust_metadata/ytbb_class_index.json').resolve()) as f:
        ytbb_class_index = json.loads(f.read())

    rev_ytbb_class_index = dict([(y, int(x)) for (x, y) in ytbb_class_index.items()])

    def project_imagenet_predictions_ytbb(preds, merge_op='max'):
        new_preds = np.zeros((preds.shape[0], 24))
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


    pmk_dataset = YtbbPmkDataset([anchor_annots], pmk_annots)
    anchor_labels = pmk_dataset.get_anchor_labels(rev_ytbb_class_index, reviewed=True)
    logits_projected = project_imagenet_predictions_ytbb(logits.numpy(), merge_op)
    pmk_frames = pmk_dataset.get_pmk(k=10, reviewed=True)
    # Collect predictions by key
    valid_pmk_keys = {key for x in pmk_frames.values() for key in x.values()}
    valid_anchor_keys = set(anchor_labels.keys())
    valid_keys = valid_pmk_keys | valid_anchor_keys
    predictions_by_key = {
        k: p
        for k, p in zip(image_paths, logits_projected) if k in valid_keys
    }


    pmk_accuracy, score_info = create_pmk_score(predictions_by_key,
                                                anchor_labels, pmk_frames)


    pm0_frames = pmk_dataset.get_pmk(k=0, reviewed=True)
    valid_pm0_keys = {key for x in pm0_frames.values() for key in x.values()}
    valid_pm0_keys = valid_pm0_keys | valid_anchor_keys
    predictions_by_key_pm0 = {
        k: p
        for k, p in zip(image_paths, logits_projected) if k in valid_pm0_keys
    }
    pm0_accuracy, score_info_pm0 = create_pmk_score(predictions_by_key_pm0,
                                                anchor_labels, pm0_frames)

    metrics_dict = {}
    metrics_dict['pm0'] = pm0_accuracy
    metrics_dict['pm10'] = pmk_accuracy
    metrics_dict['merge_op'] = merge_op
    return metrics_dict


def validate_val_on_ytbb_robust_classes(logits, targets, image_paths, py_model, merge_op='max'):
    if 'smoothing' in py_model.name:
        merge_op = 'sum'

    with open((pathlib.Path(__file__).parent / 'ytbb-robust_metadata/class_idx_map.json').resolve()) as f:
        CLASS_IDX_LIST = sorted(list(map(int, json.loads(f.read()).keys())))

    with open((pathlib.Path(__file__).parent / 'ytbb-robust_metadata/class_idx_map.json').resolve()) as f:
        CLASS_IDX_MAP = {int(k): v for k, v in json.loads(f.read()).items()}

    with open((pathlib.Path(__file__).parent / 'ytbb-robust_metadata/rev_class_idx_map.json').resolve()) as f:
        REV_CLASS_IDX_MAP = json.loads(f.read())

    anchor_annots = (pathlib.Path(__file__).parent / 'ytbb-robust_metadata/anchor_labels.json').resolve()
    pmk_annots = (pathlib.Path(__file__).parent / 'ytbb-robust_metadata/pmk_labels.json').resolve()
    pmk_dataset = YtbbPmkDataset([anchor_annots], pmk_annots)
    with open((pathlib.Path(__file__).parent / 'ytbb-robust_metadata/ytbb_class_index.json').resolve()) as f:
        ytbb_class_index = json.loads(f.read())

    rev_ytbb_class_index = dict([(y, int(x)) for (x, y) in ytbb_class_index.items()])
    anchor_labels = pmk_dataset.get_anchor_labels(rev_ytbb_class_index, reviewed=True)




    def project_imagenet_predictions_ytbb(preds, merge_op='max'):
        new_preds = np.zeros((preds.shape[0], 24))
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
        targets = np.array([CLASS_IDX_MAP[x] for x in targets])
        preds = logits_projected.argmax(axis=1)

        acc = np.mean(np.equal(preds, targets))

        weights = np.zeros((24,))
        for labels in anchor_labels.values():
            for label in labels:
                weights[int(label)] += 1

        weights /= weights.sum()
        indv_acc = np.zeros((24,))

        for i in range(24):
            if np.sum(targets == i) > 0:
                indv_acc[i] = np.mean(np.equal(preds[np.equal(targets, i)], i))

        return {'top1': acc * 100,
                'top1_uniform_class_weight': indv_acc.mean() * 100,
                'top1_ytbb_robust_balanced_class_weight': np.sum(indv_acc * weights) * 100}

    logits_projected = project_imagenet_predictions_ytbb(logits.numpy(), merge_op)

    metrics_dict = score_predictions(logits_projected, targets.numpy())
    metrics_dict['merge_op'] = merge_op
    return metrics_dict


with open((pathlib.Path(__file__).parent / 'ytbb-robust_metadata/class_idx_map.json').resolve()) as f:
	class_sublist = sorted(list(map(int, json.loads(f.read()).keys())))

registry.add_eval_setting(
    EvalSetting(
        name = 'ytbb-robust',
        dataset = StandardDataset(name='ytbb-robust'),
        size = 2530,
        metrics_fn = validate_ytbb_robust,
        class_sublist = class_sublist,
    )
)


idx_subsample_list = [range(x*50, (x+1)*50) for x in class_sublist]
idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

registry.add_eval_setting(
    EvalSetting(
        name = 'val-on-ytbb-robust-classes',
        dataset = StandardDataset(name='val'),
        size = 11550,
        metrics_fn = validate_val_on_ytbb_robust_classes,
        class_sublist = class_sublist,
        idx_subsample_list = idx_subsample_list,
    )
)
