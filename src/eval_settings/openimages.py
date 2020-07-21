import csv
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import torch

from registry import registry
from eval_settings.eval_setting_base import (
    accuracy_topk, EvalSetting, StandardDataset)

METADATA = Path(__file__).parent / 'openimages_metadata'

with open(METADATA / 'ilsvrc_to_openimages.json', 'r') as f:
    ILSVRC_TO_OPENIMAGES = json.load(f)

with open(METADATA / 'openimages_test_ilsvrc_indices.json', 'r') as f:
    OPENIMAGES_TEST_ILSVRC_INDICES = json.load(f)['indices']

SORTED_ILSVRC_INDICES = sorted(ILSVRC_TO_OPENIMAGES.keys(),
                               key=lambda x: int(x))


def validate_openimages(logits, targets, image_paths):
    # Map OpenImages label to index
    OPENIMAGE_TO_IDX = {
        ILSVRC_TO_OPENIMAGES[ilsvrc_idx]: i
        for i, ilsvrc_idx in enumerate(SORTED_ILSVRC_INDICES)
    }

    with open(METADATA / 'test-annotations-human-imagelabels.csv', 'r') as f:
        reader = csv.DictReader(f)
        OPENIMAGES_TEST_LABELS = defaultdict(list)
        for row in reader:
            if int(row['Confidence']) == 1:
                label = row['LabelName']
                if label in OPENIMAGE_TO_IDX:
                    OPENIMAGES_TEST_LABELS[row['ImageID']].append(label)

    images_per_class = Counter()
    correct_per_class = Counter()
    for i, path in enumerate(image_paths):
        image_id = Path(path).stem
        target_openimage = OPENIMAGES_TEST_LABELS[image_id]
        assert len(target_openimage) == 1, (
            f'Found more than 1 label for image {image_id}')
        # labels = {
        #     OPENIMAGE_TO_IDX[x]
        #     for x in OPENIMAGES_TEST_LABELS[image_id]
        # }
        target = OPENIMAGE_TO_IDX[target_openimage[0]]
        images_per_class[target] += 1
        if logits[i, :].argmax().item() == target:
            correct_per_class[target] += 1
    balanced_accuracy = np.mean(
        [correct_per_class[c] / images_per_class[c] for c in images_per_class])
    accuracy = sum(correct_per_class.values()) / sum(images_per_class.values())

    return {'top1': accuracy, 'top1_balanced': balanced_accuracy}


class_sublist = [int(x) for x in SORTED_ILSVRC_INDICES]

registry.add_eval_setting(
    EvalSetting(
        name='openimages_test_ilsvrc_subset',
        dataset=StandardDataset(name='openimages-test'),
        idx_subsample_list=OPENIMAGES_TEST_ILSVRC_INDICES,
        size=23104,
        metrics_fn=validate_openimages,
        class_sublist=class_sublist,
    ))


idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in class_sublist]
idx_subsample_list = sorted(
    [item for sublist in idx_subsample_list for item in sublist])


def accuracy_topk_subselected(logits, targets):
    targets = torch.tensor([class_sublist.index(x) for x in targets])
    return accuracy_topk(logits, targets)


registry.add_eval_setting(
    EvalSetting(
        name='val-on-openimages-classes',
        dataset=StandardDataset(name='val'),
        size=20700,
        class_sublist=class_sublist,
        idx_subsample_list=idx_subsample_list,
        metrics_fn=accuracy_topk_subselected,
    ))
