import io
import random
from os.path import join, exists
import torch
import pickle

from mldb import model_repository
from mldb.s3_utils import untar_directory, default_cache_root_path


m_repo = model_repository.ModelRepository()

DATASET_NAMES = set(x.name for x in m_repo.get_datasets())
MODEL_NAMES = set(x.name for x in m_repo.get_models() if x.completed and not x.hidden)
EVAL_SETTING_NAMES = set(x.name for x in m_repo.get_evaluation_settings() if not x.hidden)


def evaluation_completed(py_model, py_eval_setting):
    assert py_model.name in MODEL_NAMES, \
            f'Model {py_model.name} is not recognized as an existing model in the' + \
            ' server. Did you run the db script?'
    assert py_eval_setting.name in EVAL_SETTING_NAMES, \
            f'Model {py_eval_setting.name} is not recognized as an existing eval setting in the' + \
            ' server. Did you run the db script?'

    checkpoint = m_repo.get_model(name=py_model.name, load_final_checkpoint=True, load_evaluations=True).final_checkpoint
    setting_uuids = [m_repo.get_evaluation_setting(name=py_eval_setting.name).uuid]

    if py_eval_setting.parent_eval_setting is not None:
        assert py_eval_setting.parent_eval_setting in EVAL_SETTING_NAMES, \
            f'Eval setting {py_eval_setting.parent_eval_setting} is not recognized as an existing eval setting in the' + \
            ' server. Did you run the db script?'
        
        setting_uuids += [m_repo.get_evaluation_setting(name=py_eval_setting.parent_eval_setting).uuid]

    for e in m_repo.get_evaluations([x.uuid for x in checkpoint.evaluations]):
        if e.setting_uuid in setting_uuids and e.completed:
            return True
    return False


def store_evaluation(py_model, py_eval_setting, metrics, logits):
    assert py_model.name in MODEL_NAMES, \
            f'Model {py_model.name} is not recognized as an existing model in the' + \
            ' server. Did you run the db script?'
    assert py_eval_setting.name in EVAL_SETTING_NAMES, \
            f'Model {py_eval_setting.name} is not recognized as an existing eval setting in the' + \
            ' server. Did you run the db script?'

    bio = io.BytesIO()
    torch.save(logits.cpu(), bio)

    model_uuid = m_repo.get_model(name=py_model.name).final_checkpoint_uuid
    setting_uuid = m_repo.get_evaluation_setting(name=py_eval_setting.name).uuid

    eval = m_repo.create_evaluation(checkpoint_uuid=model_uuid, setting_uuid=setting_uuid, 
                                    extra_info=metrics, logits_data_bytes=bio.getvalue(), completed=True)

def download_dataset(dataset):
    dataset = m_repo.get_dataset(name=dataset)
    filedir = join(default_cache_root_path, f'datasets/{dataset.name}')
    if not exists(filedir):
        filename = filedir + '.tar'
        m_repo.download_dataset_data(dataset_uuid=dataset.uuid, target_filename=filename)

        if 'format-val' in dataset.name:
            strip, one_top_level = 2, False
        elif 'imagenet-c' in dataset.name:
            strip, one_top_level = 6, True
        elif dataset.name in ['imagenetv2-matched-frequency', 'imagenetv2-topimages', 'imagenetv2-threshold0.7', 'val']:
            strip, one_top_level = 3, False
        else:
            strip, one_top_level = 1, True
        untar_directory(filename, join(default_cache_root_path, 'datasets'), strip=strip, one_top_level=one_top_level)

    return filedir


def load_model_checkpoint_bytes(model_name):
    r_model = m_repo.get_model(name=model_name)
    data = m_repo.get_checkpoint_data(r_model.final_checkpoint_uuid)
    bio = io.BytesIO(data)
    return bio   


def load_model_state_dict(model, name):
    bio = load_model_checkpoint_bytes(name)
    state_dict = torch.load(bio, map_location=f'cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)


def add_model_shell(model_name):
    r_model = m_repo.create_model(name=model_name, completed=True)
    checkpoint = m_repo.create_checkpoint(model_uuid=r_model.uuid)
    m_repo.set_final_model_checkpoint(r_model.uuid, checkpoint.uuid)


def hide_rename_model(model_name):
    model = m_repo.get_model(name=model_name, load_final_checkpoint=True, load_evaluations=True)
    for e in m_repo.get_evaluations([x.uuid for x in model.final_checkpoint.evaluations]):
        m_repo.hide_evaluation(e.uuid)
    new_name = model_name + f'_hidden_{random.randint(0, 10000)}'
    m_repo.rename_model(model.uuid, new_name)
    m_repo.hide_model(model.uuid)


def create_eval_setting(eval_setting_name):
    m_repo.create_evaluation_setting(name=eval_setting_name)


def hide_rename_eval_setting(eval_setting_name):
    setting = m_repo.get_evaluation_setting(name=eval_setting_name, load_evaluations=True)
    for e in m_repo.get_evaluations([x.uuid for x in setting.evaluations]):
        m_repo.hide_evaluation(e.uuid)
    new_name = eval_setting_name + f'_hidden_{random.randint(0, 10000)}'
    m_repo.rename_evaluation_setting(setting.uuid, new_name)
    m_repo.hide_evaluation_setting(setting.uuid)


def rename_model(model_name, new_model_name):
    uuid = m_repo.get_model(name=model_name).uuid
    m_repo.rename_model(uuid, new_model_name)


def rename_eval_setting(eval_setting_name, new_eval_setting_name):
    uuid = m_repo.get_evaluation_setting(name=eval_setting_name).uuid
    m_repo.rename_evaluation_setting(uuid, new_eval_setting_name)


def hide_evaluation(model_name, eval_setting_name):
    flag = False
    for e in m_repo.get_evaluations():
        if e.checkpoint.model.name == model_name and e.setting.name == eval_setting_name:
            m_repo.hide_evaluation(e.uuid)
            flag = True
    return flag


def get_eval_extra_info(model_name, eval_setting_name):
    for e in m_repo.get_evaluations():
        if e.checkpoint.model.name == model_name and e.setting.name == eval_setting_name and e.completed:
            return e.extra_info


def close_db_connection():
    m_repo.dispose()
