import os
import sys
import pathlib
import copy
import re

import pandas as pd
import numpy as np

sys.path.append(str((pathlib.Path(__file__).parent / '../src').resolve()))

from mldb.utils import m_repo
from mldb.model_repository import Model, EvaluationSetting
from registry import registry
registry.load_full_registry()

def unique_insert(d, key, value, name):
    if key in d:
        print(f'Key {key} for name {name} already appears in the dictionary')
    assert key not in d 
    d[key] = value


def is_eval_invalid(eval):
    return eval.hidden or not eval.completed or eval.checkpoint.model.hidden \
           or not eval.checkpoint.model.completed or eval.setting.hidden


def download_grid_dataframe():
    rows = {}
    for eval in m_repo.get_evaluations():
        if is_eval_invalid(eval):
            continue

        model = eval.checkpoint.model
        if model not in rows:
            rows[model] = {'model': model}
        row = rows[model]

        if 'imagenet-vid-robust' == eval.setting.name or 'ytbb-robust' == eval.setting.name:
            for pm in [0, 10]:
                setting = copy.copy(eval.setting)
                setting.name += f'_pm{pm}'
                unique_insert(row, setting, eval.extra_info[f'pm{pm}'] * 100, model.name)
        elif 'openimages_test' in eval.setting.name:
            unique_insert(row, eval.setting, eval.extra_info['top1']*100, model.name)            
        else:
            unique_insert(row, eval.setting, eval.extra_info['top1'], model.name)

    df = pd.DataFrame(rows.values())
    df = df.sort_values(by=m_repo.get_evaluation_setting(name='val'), ascending=False)
    return df


def strip_metadata(df):
    df.model = [x.name for x in df.model]
    df = df.set_index('model')
    df.columns = [x.name if type(x) == EvaluationSetting else x for x in df.columns]
    return df


def replace_10percent_naive(df):
    linked_suffix = '_10percent'
    for col in [x for x in df.columns if linked_suffix in x]:
        linked_col = col.replace(linked_suffix, '')
        if linked_col not in df.columns:
            df[linked_col] = np.nan
        df[linked_col].fillna(df[col], inplace=True)
        df = df.drop(columns=col)
    return df


def aggregate_corruptions_naive(df):
    df_short = df.copy()
    corrs = set(x.split('.')[1]+'_'+x.split('_')[-1] for x in df_short.columns if 'imagenet-c' in x)
    for corr in corrs:
        c_name, c_type = corr.rsplit('_', 1)
        col_match = df_short.columns.str.match(f'imagenet-c.{c_name}.*_{c_type}')
        new_col = df_short.loc[:, col_match].mean(axis=1, skipna=False)
        df_short = df_short.drop(columns=df_short.columns[col_match])
        df_short[f'{c_name}_{c_type}'] = new_col

    return df_short


def extract_metadata(df):
    df_metadata = df.copy()
    df_metadata['arch'] = [registry.get_model(x.name).arch if registry.contains_model(x.name) else 'N/A' for x in df.model]

    for col in df_metadata.columns:
        if type(col) == EvaluationSetting:
            size = registry.get_eval_setting(col.name.replace('_pm0', '').replace('_pm10', '')).size
            df_metadata[col.name+'_dataset_size'] = [size] * df_metadata.shape[0]
            df_metadata = df_metadata.drop(columns=col)

    df, df_metadata = strip_metadata(df), strip_metadata(df_metadata)
    return df, df_metadata


def replace_10percent_with_metadata(df, df_metadata):
    for col in [x for x in df.columns if '_10percent' in x]:
        linked_col = col.replace('_10percent', '')

        if linked_col not in df.columns:
            df[linked_col] = np.nan
            df_metadata[linked_col+'_dataset_size'] = np.nan

        df_metadata.loc[df[linked_col].isnull(), linked_col+'_dataset_size'] = np.nan
        df_metadata.loc[df[col].isnull(), col+'_dataset_size'] = np.nan

        df[linked_col].fillna(df[col], inplace=True)
        df_metadata[linked_col+'_dataset_size'].fillna(df_metadata[col+'_dataset_size'], inplace=True)
        df = df.drop(columns=col)
        df_metadata = df_metadata.drop(columns=col+'_dataset_size')

    return df, df_metadata


def aggregate_corruptions_with_metadata(df, df_metadata):
    corrs = set(x.split('.')[1]+'_'+x.split('_')[-1] for x in df.columns if 'imagenet-c' in x)
    for corr in corrs:
        c_name, c_type = corr.rsplit('_', 1)

        col_match = df.columns.str.match(f'imagenet-c.{c_name}.*_{c_type}')
        new_col = df.loc[:, col_match].mean(axis=1, skipna=False)
        df = df.drop(columns=df.columns[col_match])
        df[f'{c_name}_{c_type}'] = new_col

        col_match = df_metadata.columns.str.match(f'imagenet-c.{c_name}.*_{c_type}_dataset_size')
        new_col = df_metadata.loc[:, col_match].max(axis=1, skipna=False)
        df_metadata = df_metadata.drop(columns=df_metadata.columns[col_match])
        df_metadata[f'{c_name}_{c_type}_dataset_size'] = new_col

    match = [x for x in df.columns if any(y in x for y in ['disk', 'memory', 'stylized', 'greyscale'])]
    df['avg_corruptions'] = df[match].mean(axis=1, skipna=False)
    match = [x for x in df_metadata.columns if any(y in x for y in ['disk', 'memory', 'stylized', 'greyscale'])]
    df_metadata['avg_corruptions_dataset_size'] = df_metadata[match].max(axis=1, skipna=False)

    match = [x for x in df.columns if 'pgd' in x]
    df['avg_pgd'] = df[match].mean(axis=1, skipna=False)
    match = [x for x in df_metadata.columns if 'pgd' in x]
    df_metadata['avg_pgd_dataset_size'] = df_metadata[match].max(axis=1, skipna=False)

    return df, df_metadata


def add_aggregate_corruptions_for_plotting(df, df_metadata):
    corrs = set(x.split('.')[1]+'_'+x.split('_')[-1] for x in df.columns if 'imagenet-c' in x)
    for corr in corrs:
        c_name, c_type = corr.rsplit('_', 1)

        col_match = df.columns.str.match(f'imagenet-c.{c_name}.*_{c_type}')
        new_col = df.loc[:, col_match].mean(axis=1, skipna=False)
        df[f'{c_name}_{c_type}'] = new_col

        col_match = df_metadata.columns.str.match(f'imagenet-c.{c_name}.*_{c_type}_dataset_size')
        new_col = df_metadata.loc[:, col_match].max(axis=1, skipna=False)
        df_metadata[f'{c_name}_{c_type}_dataset_size'] = new_col

    match = [x for x in df.columns if any(y in x for y in ['disk', 'memory', 'stylized', 'greyscale']) and not re.search('\d', x)]
    df['avg_corruptions'] = df[match].mean(axis=1, skipna=False)
    match = [x for x in df_metadata.columns if any(y in x for y in ['disk', 'memory', 'stylized', 'greyscale']) and not re.search('\d', x)]
    df_metadata['avg_corruptions_dataset_size'] = df_metadata[match].max(axis=1, skipna=False)

    match = [x for x in df.columns if 'memory' in x and not re.search('\d', x)]
    df['avg_imagenet_c_in_memory'] = df[match].mean(axis=1, skipna=False)
    match = [x for x in df_metadata.columns if 'memory' in x and not re.search('\d', x)]
    df_metadata['avg_imagenet_c_in_memory_dataset_size'] = df_metadata[match].max(axis=1, skipna=False)

    match = [x for x in df.columns if 'disk' in x and not re.search('\d', x)]
    df['avg_imagenet_c_on_disk'] = df[match].mean(axis=1, skipna=False)
    match = [x for x in df_metadata.columns if 'disk' in x and not re.search('\d', x)]
    df_metadata['avg_imagenet_c_on_disk_dataset_size'] = df_metadata[match].max(axis=1, skipna=False)

    match = [x for x in df.columns if 'pgd' in x]
    df['avg_pgd'] = df[match].mean(axis=1, skipna=False)
    match = [x for x in df_metadata.columns if 'pgd' in x]
    df_metadata['avg_pgd_dataset_size'] = df_metadata[match].max(axis=1, skipna=False)

    return df, df_metadata
