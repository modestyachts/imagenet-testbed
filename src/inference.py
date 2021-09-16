import argparse
import json
from os.path import join
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import CustomImageFolder, DistributedSampler
from registry import registry
from models.low_accuracy import TimmFeatures, IdentityFeatures, SklearnCLF, Simple1NN


def main(args):
    py_model = registry.get_model(args.model)
    py_eval_setting = registry.get_eval_setting(args.eval_setting)

    if args.db and utils.evaluation_completed(py_model, py_eval_setting):
        print(f'Evaluation for {py_model.name} x {py_eval_setting.name} already found. Skipping...')
        return

    args.num_gpus = torch.cuda.device_count()
    results_dict = mp.Manager().dict()
    mp.spawn(main_worker, nprocs=args.num_gpus, args=(args, results_dict))

    idx_sorted, idx_map = torch.cat([results_dict[i]['idxs'] for i in range(args.num_gpus)]).sort()
    assert idx_sorted.eq(idx_sorted.unique()).all(), 'Error collecting results'
    assert idx_sorted.eq(torch.tensor(list(range(idx_sorted.size(0))))).all(), 'Error collecting results'

    logits = torch.cat([results_dict[i]['logits'] for i in range(args.num_gpus)])[idx_map]
    targets = torch.cat([results_dict[i]['targets'] for i in range(args.num_gpus)])[idx_map]
    image_paths = np.concatenate([results_dict[i]['image_paths'] for i in range(args.num_gpus)])[idx_map]

    metrics = py_eval_setting.get_metrics(logits, targets, image_paths, py_model)

    with open(join(args.logdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    torch.save(logits, join(args.logdir, 'logits.pt'))
    torch.save(targets, join(args.logdir, 'targets.pt'))
    with open(join(args.logdir, 'image_paths.pkl'), 'wb') as f:
        np.save(f, image_paths)

    if args.db:
        utils.store_evaluation(py_model, py_eval_setting, metrics, logits)
        print('Uploaded to db')
        utils.close_db_connection()

    print('************************************')
    print(f'RESULT {args.model} on {args.eval_setting} - {metrics}')
    print('************************************')


def main_worker(gpu, args, results_dict):
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    registry.load_full_registry()
    py_model = registry.get_model(args.model)
    py_eval_setting = registry.get_eval_setting(args.eval_setting)

    model = py_model.generate_classifier(py_eval_setting)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    batch_size = py_model.get_batch_size(py_eval_setting)
    gpu_perturbation_fn = py_eval_setting.get_perturbation_fn_gpu(py_model)
    torch.set_grad_enabled(py_eval_setting.adversarial_attack is not None)

    setting_transform = [py_eval_setting.transform] if py_eval_setting.transform is not None else []
    val_dataset = CustomImageFolder(
        root = py_eval_setting.get_dataset_root(),
        transform = transforms.Compose(setting_transform + [py_model.transform]),
        perturbation_fn = py_eval_setting.get_perturbation_fn_cpu(py_model),
        idx_subsample_list = py_eval_setting.get_idx_subsample_list(py_model),
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=args.num_gpus, rank=gpu, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=False, sampler=val_sampler)
    
    logits, targets, image_paths, idxs = validate(gpu, args, val_loader, model, gpu_perturbation_fn)
    results_dict[gpu] = {'logits': logits, 'targets': targets, 'image_paths': image_paths, 'idxs': idxs}


def validate(gpu, args, val_loader, model, gpu_perturbation_fn):
    model.eval()
    all_logits, all_targets, all_idxs, all_image_paths = [], [], [], []

    if gpu == 0:
        val_loader = tqdm(val_loader, desc='Validating')

    for idxs, image_paths, images, target in val_loader:
        images = images.cuda()

        if gpu_perturbation_fn is not None:
            images = gpu_perturbation_fn(images, model)
        output = model(images)

        all_logits.append(output.detach().cpu())
        all_targets.append(target)
        all_idxs.append(idxs)
        all_image_paths.append(image_paths)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_idxs = torch.cat(all_idxs, dim=0)
    all_image_paths = [image_path for batch in all_image_paths for image_path in batch]

    return all_logits, all_targets, all_image_paths, all_idxs


if __name__ == '__main__':
    from mldb import utils
    
    registry.load_full_registry()

    parser = argparse.ArgumentParser(description='ML Robustness Evaluation')
    parser.add_argument('--model', choices=registry.model_names(),
                        help='name of the model to evaluate')
    parser.add_argument('--eval-setting', choices=registry.eval_setting_names(),
                        help='evaluation setting to run the model on')
    parser.add_argument('--db', action='store_true',
                        help='specify flag to store results in database')
    parser.add_argument('--workers', type=int, default=4, 
                        help='number of data loading workers')
    parser.add_argument('--logdir', type=str, metavar='LOGDIR',
                        help='path to log directory')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='gpu communication backend')
    parser.add_argument('--dist-url', type=str, default=f'tcp://127.0.0.1:{np.random.randint(20000, 30000)}',
                        help='distributed training coordination initialization')
    args = parser.parse_args()

    main(args)
