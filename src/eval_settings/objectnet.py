import json
import torch
from pathlib import Path
import PIL

from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset, accuracy_topk


METADATA = Path(__file__).parent / 'objectnet_metadata'

with open(METADATA / 'folder_to_objectnet_label.json', 'r') as f:
	folder_map = json.load(f)
	folder_map = {v: k for k, v in folder_map.items()}

with open(METADATA / 'objectnet_to_imagenet_1k.json', 'r') as f:
	objectnet_map = json.load(f)

with open(METADATA / 'imagenet_to_labels.json', 'r') as f:
	imagenet_map = json.load(f)
	imagenet_map = {v: k for k, v in imagenet_map.items()}


folder_to_ids, class_sublist = {}, []
for objectnet_name, imagenet_names in objectnet_map.items():
	imagenet_names = imagenet_names.split('; ')
	imagenet_ids = [int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names]
	class_sublist.extend(imagenet_ids)
	folder_to_ids[folder_map[objectnet_name]] = imagenet_ids


def crop_image(image, border=2):
	return PIL.ImageOps.crop(image, border=border)


def objectnet_accuracy(logits, targets, image_paths, using_class_sublist=True):
	if using_class_sublist:
		folder_map = {k: [class_sublist.index(x) for x in v] for k, v in folder_to_ids.items()}
	else:
		folder_map = folder_to_ids

	preds = logits.argmax(dim=1)
	num_correct, num_total = 0, 0
	for pred, image_path in zip(preds, image_paths):
		folder = image_path.split('/')[0]
		if folder in folder_map:
			num_total += 1
			if pred in folder_map[folder]:
				num_correct += 1
	return {'top1': num_correct / num_total * 100}


registry.add_eval_setting(
    EvalSetting(
        name = 'objectnet-1.0-beta',
        dataset = StandardDataset(name='objectnet-1.0-beta'),
        size = 18514,
        class_sublist = class_sublist,
        metrics_fn = objectnet_accuracy,
        transform = crop_image,
    )
)


idx_subsample_list = [range(x*50, (x+1)*50) for x in class_sublist]
idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

folder_map_sublisted = {k: [class_sublist.index(x) for x in v] for k, v in folder_to_ids.items()}
objectnet_idx_to_imgnet_idxs = {idx: folder_map_sublisted[name] for idx, name in enumerate(sorted(folder_map_sublisted))}
imgnet_idx_to_objectnet_idx =  {}
for objectnet_idx, imgnet_idxs in objectnet_idx_to_imgnet_idxs.items():
	for imgnet_idx in imgnet_idxs:
		imgnet_idx_to_objectnet_idx[imgnet_idx] = objectnet_idx

def accuracy_topk_subselected_and_collapsed(logits, targets):
    collapsed_logits = torch.zeros((logits.size(0), len(folder_map_sublisted)), dtype=logits.dtype, device=logits.device)
    for objectnet_idx, imgnet_idxs in objectnet_idx_to_imgnet_idxs.items():
        collapsed_logits[:, objectnet_idx] = logits[:, imgnet_idxs].max(dim=1).values
    targets = torch.tensor([imgnet_idx_to_objectnet_idx[class_sublist.index(x)] for x in targets])
    return accuracy_topk(collapsed_logits, targets)

registry.add_eval_setting(
    EvalSetting(
        name = 'val-on-objectnet-classes',
        dataset = StandardDataset(name='val'),
        size = 6250,
        class_sublist = class_sublist,
        idx_subsample_list = idx_subsample_list,
        metrics_fn = accuracy_topk_subselected_and_collapsed,
    )
)
