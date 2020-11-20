import argparse
from tqdm import tqdm
import pickle
import torch
import io
from os.path import join

import sys
sys.path.append('src/')
from mldb import utils
from registry import registry
registry.load_full_registry()


parser = argparse.ArgumentParser(description='ML Robustness Evaluation')
parser.add_argument('--add-model', type=str, metavar='model',
                    help='name of the model to add to the db')
parser.add_argument('--remove-model', type=str, metavar='model',
                    help='name of the model to remove from the db')

parser.add_argument('--add-eval-setting', type=str, metavar='eval-setting',
                    help='name of the eval setting to add to the db')
parser.add_argument('--remove-eval-setting', type=str, metavar='eval-setting',
                    help='name of the eval setting to remove from the db')

parser.add_argument('--remove-evaluation', type=str, nargs=2, metavar=('model', 'eval-setting'),
                    help='model and eval_setting for which evaluation should be removed')

parser.add_argument('--rename-model', type=str, nargs=2, metavar=('current-name', 'new-name'),
                    help='current name and new name for the model for which to rename')
parser.add_argument('--rename-eval-setting', type=str, nargs=2, metavar=('current-name', 'new-name'),
                    help='current name and new name for the eval setting for which to rename')

parser.add_argument('--print-eval', type=str, nargs=2, metavar=('model', 'eval-setting'),
                    help='print evaluation for model on eval setting')

parser.add_argument('--list-models-db', action='store_true',
                    help='lists models in the db')
parser.add_argument('--list-models-registry', action='store_true',
                    help='lists models in the registry')

parser.add_argument('--list-eval-settings-db', action='store_true',
                    help='lists eval settings in the db')
parser.add_argument('--list-eval-settings-registry', action='store_true',
                    help='lists eval settings in the registry')
parser.add_argument('--list-parent-eval-settings-registry', action='store_true',
                    help='lists non-child eval settings in the registry')

parser.add_argument('--save-logits', type=str, nargs=2, metavar=('eval-setting', 'logdir'),
                    help='save prediction logits from all models on eval-setting to logdir')

parser.add_argument('--override', action='store_true',
                    help='safeguard against potentially drastic actions')

args = parser.parse_args()


if args.add_model:
    assert args.add_model not in utils.MODEL_NAMES, f'Model with name {args.add_model} already exists in db'

    if registry.contains_model(args.add_model) or args.override:
        utils.add_model_shell(args.add_model)
        print(f'Model {args.add_model} added to db')
    
    else:
        print(f'Model {args.add_model} implementation not found in registry. Specify --override to add to db anyway')


if args.remove_model:
    assert args.remove_model in utils.MODEL_NAMES, f'Model {args.remove_model} does not exist in db'

    if args.override:
        utils.hide_rename_model(args.remove_model)
        print(f'Model {args.remove_model} removed from db')
    
    else:
        print(f'NOTE: Removing model {args.remove_model} will remove all associated evaluations. Specify --override to remove anyway')


if args.add_eval_setting:
    assert args.add_eval_setting not in utils.EVAL_SETTING_NAMES, f'Eval setting with name {args.eval_setting} already exists in db'

    if registry.contains_eval_setting(args.add_eval_setting) or args.override:
        utils.create_eval_setting(args.add_eval_setting)
        print(f'Eval setting {args.add_eval_setting} added to db')
    
    else:
        print(f'Eval setting {args.add_eval_setting} implementation not found in registry. Specify --override to add to db anyway')


if args.remove_eval_setting:
    assert args.remove_eval_setting in utils.EVAL_SETTING_NAMES, f'Eval setting with name {args.remove_eval_setting} does not exist in db'

    if args.override:
        utils.hide_rename_eval_setting(args.remove_eval_setting)
        print(f'Eval setting {args.remove_eval_setting} removed from db')
    
    else:
        print(f'NOTE: Removing eval setting {args.remove_eval_setting} will remove all associated evaluations. Specify --override to remove anyway')


if args.remove_evaluation:
    assert args.remove_evaluation[0] in utils.MODEL_NAMES, f'Model {args.remove_evaluation[0]} does not exist in db'
    assert args.remove_evaluation[1] in utils.EVAL_SETTING_NAMES, f'Eval setting {args.remove_evaluation[1]} does not exist in db'

    result = utils.hide_evaluation(args.remove_evaluation[0], args.remove_evaluation[1])
    if result:
        print(f'Evaluation for {args.remove_evaluation[0]} on {args.remove_evaluation[1]} removed')
    else:
        print(f'Evaluation for {args.remove_evaluation[0]} on {args.remove_evaluation[1]} not found in db')


if args.rename_model:
    assert args.rename_model[0] in utils.MODEL_NAMES, f'Model {args.rename_model[0]} does not exist in db'
    assert args.rename_model[1] not in utils.MODEL_NAMES, f'Model name {args.rename_model[1]} already exists in db'

    utils.rename_model(args.rename_model[0], args.rename_model[1])
    print(f'Model {args.rename_model[0]} renamed to {args.rename_model[1]} in db. Make sure your definition file also has this change.')


if args.rename_eval_setting:
    assert args.rename_eval_setting[0] in utils.EVAL_SETTING_NAMES, f'Eval setting {args.rename_eval_setting[0]} does not exist in db'
    assert args.rename_eval_setting[1] not in utils.EVAL_SETTING_NAMES, f'Eval setting name {args.rename_eval_setting[1]} already exists in db'

    utils.rename_eval_setting(args.rename_eval_setting[0], args.rename_eval_setting[1])
    print(f'Eval setting {args.rename_eval_setting[0]} renamed to {args.rename_eval_setting[1]} in db. Make sure to also update your definition file.')


if args.print_eval:
    assert args.print_eval[0] in utils.MODEL_NAMES, f'Model {args.print_eval[0]} does not exist in db'
    assert args.print_eval[1] in utils.EVAL_SETTING_NAMES, f'Eval setting {args.print_eval[1]} does not exist in db'

    evaluation = utils.get_eval_extra_info(args.print_eval[0], args.print_eval[1])
    print(f'Evaluation for {args.print_eval[0]} on {args.print_eval[1]} - {evaluation}')


if args.list_models_db:
    print('\nMODELS IN DB:\n' + ' '.join(sorted(list(utils.MODEL_NAMES))))

if args.list_models_registry:
    print('\nMODELS IN REGISTRY:\n' + ' '.join(sorted(list(registry.model_names()))))

if args.list_eval_settings_db:
    print('\nEVAL SETTINGS IN DB:\n' + ' '.join(sorted(list(utils.EVAL_SETTING_NAMES))))

if args.list_eval_settings_registry:
    print('\nEVAL SETTINGS IN REGISTRY:\n' + ' '.join(sorted(list(registry.eval_setting_names()))))

if args.list_parent_eval_settings_registry:
    names = [name for name, setting in registry.eval_settings.items() if setting.parent_eval_setting is None]
    print('\nPARENT EVAL SETTINGS IN REGISTRY:\n' + ' '.join(sorted(names)))


if args.save_logits:
    assert args.save_logits[0] in utils.EVAL_SETTING_NAMES, f'Eval setting {args.save_logits[0]} does not exist in db'

    py_eval_setting = utils.m_repo.get_evaluation_setting(name=args.save_logits[0], load_evaluations=True)
    evals = utils.m_repo.get_evaluations([x.uuid for x in py_eval_setting.evaluations])

    for eval in tqdm(evals):
        if not utils.m_repo.has_evaluation_logits_data(eval.uuid):
            print(f'Evaluation logits for model {eval.checkpoint.model.name} not found.')
            continue

        data = utils.m_repo.get_evaluation_logits_data(eval.uuid)
        logits = torch.load(io.BytesIO(data)).numpy()
        logits_dict = {'model': eval.checkpoint.model.name, 'eval-setting': args.save_logits[0], 'logits': logits}

        filename = join(args.save_logits[1], f'{args.save_logits[0]}_|_{eval.checkpoint.model.name.replace("/", "_")}.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(logits_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
