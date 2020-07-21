import subprocess
import argparse
import os
from os.path import join
import time
import datetime

import sys
sys.path.append('src/')
from registry import registry
registry.load_full_registry()


parser = argparse.ArgumentParser(description='ML Robustness Evaluation')
parser.add_argument('--models', choices=registry.model_names(), nargs='+',
                    help='names of the models to evaluate')
parser.add_argument('--eval-settings', choices=registry.eval_setting_names(), nargs='+',
                    help='evaluation settings to run the models on')
parser.add_argument('--db', action='store_true',
                    help='stores results in the database')
parser.add_argument('--gpus', type=int, nargs='+',
                    help='which gpus to run on')
parser.add_argument('--logdir', type=str, default='./outputs/',
                    help=f'path to log dir')
args = parser.parse_args()


assert args.models is not None, 'Please specify at least one model'
assert args.eval_settings is not None, 'Please specify at least one evaluation setting'
assert args.gpus is not None, 'Please specify at least one gpu to run on'


for model in args.models:
    for eval_setting in args.eval_settings:

        # download dataset (if necessary)
        registry.get_eval_setting(eval_setting).get_dataset_root()

        dt = datetime.datetime.fromtimestamp(time.time())
        logdir = join(args.logdir, dt.strftime('%Y-%m-%d_%H:%M:%S'))

        cmd_params = f'--model={model} --eval-setting={eval_setting} {"--db" if args.db else ""} --logdir={logdir}'
        cmd = f'CUDA_VISIBLE_DEVICES={",".join(map(str, args.gpus))} {sys.executable} src/inference.py {cmd_params}'

        print(f'Logging to {logdir}')
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        print(cmd)

        subprocess.run(f'echo {cmd} > {logdir}/task.cmd', shell=True)
        subprocess.run(cmd, shell=True)
