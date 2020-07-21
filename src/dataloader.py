import os
import sys
import math
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CustomImageFolder(VisionDataset):

    def __init__(self, root, transform, perturbation_fn=None, idx_subsample_list=None):
        super().__init__(root, transform=transform, target_transform=None)

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.root))

        if idx_subsample_list is not None:
            samples = [samples[i] for i in idx_subsample_list]

        self.loader = pil_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.perturbation_fn = perturbation_fn

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        return [index, os.path.relpath(path, self.root), self.load_img(path), target]

    def load_img(self, path):
        sample = self.loader(path)
        sample = self.transform(sample)
        if self.perturbation_fn is not None:
            sample = self.perturbation_fn(sample)
        return sample

    def __len__(self):
        return len(self.samples)


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # subsample
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
