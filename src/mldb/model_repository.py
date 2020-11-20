import contextlib
from datetime import datetime, timezone
import getpass
import io
import json
import pathlib
import uuid
import pickle
import hashlib
import subprocess
from os.path import join, exists

import numpy as np
import sqlalchemy as sqla
from sqlalchemy.ext.declarative import declarative_base as sqla_declarative_base
from sqlalchemy_utils import database_exists, create_database

from . import s3_utils


sqlalchemy_base = sqla_declarative_base()


DB_DUMP_URL = 'https://robustness-eval.s3-us-west-2.amazonaws.com/robustness_evaluation.db'


def download_db():
    if not exists(join(s3_utils.default_cache_root_path, 'robustness_evaluation.db')):
        print('downloading database dump...')
        subprocess.run(['wget', '-P', s3_utils.default_cache_root_path, DB_DUMP_URL], check=True)


def gen_short_uuid(num_chars=None):
    num = uuid.uuid4().int
    alphabet = '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    res = []
    while num > 0:
        num, digit = divmod(num, len(alphabet))
        res.append(alphabet[digit])
    res2 = ''.join(reversed(res))
    if num_chars is None:
        return res2
    else:
        return res2[:num_chars]


def get_logdir_key(model_id):
    return 'logdir/{}'.format(model_id)


def get_checkpoint_data_key(checkpoint_id):
    return 'checkpoints/{}_data.bytes'.format(checkpoint_id)


def get_dataset_data_key(dataset_id):
    return 'datasets/{}_data.bytes'.format(dataset_id)


def get_evaluation_setting_extra_data_key(evaluation_setting_id):
    return 'evaluation_settings/{}_extra_data.bytes'.format(evaluation_setting_id)


def get_evaluation_setting_processed_dataset_key(evaluation_setting_id):
    return 'evaluation_settings/{}_processed_dataset.bytes'.format(evaluation_setting_id)


def get_raw_input_data_key(raw_input_id):
    return 'raw_inputs/{}_data.bytes'.format(raw_input_id)


def get_evaluation_extra_data_key(evaluation_id):
    return 'evaluations/{}_data.bytes'.format(evaluation_id)


def get_evaluation_logits_data_key(evaluation_id):
    return 'evaluations/{}_logits_data.bytes'.format(evaluation_id)


def get_evaluation_chunk_extra_data_key(evaluation_chunk_id):
    return 'evaluation_chunks/{}_data.bytes'.format(evaluation_chunk_id)


def get_evaluation_chunk_logits_data_key(evaluation_chunk_id):
    return 'evaluation_chunks/{}_logits_data.bytes'.format(evaluation_chunk_id)


def get_evaluation_chunk_indices_data_key(evaluation_chunk_id):
    return 'evaluation_chunks/{}_indices_data.bytes'.format(evaluation_chunk_id)


class Model(sqlalchemy_base):
    __tablename__ = 'models'
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True)
    description = sqla.Column(sqla.String)
    username = sqla.Column(sqla.String)
    creation_time = sqla.Column(sqla.DateTime(timezone=False), server_default=sqla.sql.func.now())
    extra_info = sqla.Column(sqla.JSON)
    checkpoints = sqla.orm.relationship('Checkpoint', back_populates='model', cascade='all, delete, delete-orphan', foreign_keys='Checkpoint.model_uuid')
    final_checkpoint_uuid = sqla.Column(sqla.String, sqla.ForeignKey('checkpoints.uuid'), nullable=True)
    final_checkpoint = sqla.orm.relationship('Checkpoint', foreign_keys=[final_checkpoint_uuid], uselist=False)
    completed = sqla.Column(sqla.Boolean)
    hidden = sqla.Column(sqla.Boolean)
    logdir_filepaths = sqla.Column(sqla.JSON)

    def __repr__(self):
        return f'<Model(uuid="{self.uuid}", name="{self.name}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class Checkpoint(sqlalchemy_base):
    __tablename__ = 'checkpoints'
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True)
    creation_time = sqla.Column(sqla.DateTime(timezone=False), server_default=sqla.sql.func.now())
    model_uuid = sqla.Column(sqla.String, sqla.ForeignKey('models.uuid'), nullable=False)
    model = sqla.orm.relationship('Model', back_populates='checkpoints', foreign_keys=[model_uuid])
    evaluations = sqla.orm.relationship('Evaluation', back_populates='checkpoint', cascade='all, delete, delete-orphan', foreign_keys='Evaluation.checkpoint_uuid')
    training_step = sqla.Column(sqla.BigInteger)
    epoch = sqla.Column(sqla.Float)
    username = sqla.Column(sqla.String)
    extra_info = sqla.Column(sqla.JSON)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<Checkpoint(uuid="{self.uuid}", model_uuid="{self.model_uuid}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class Dataset(sqlalchemy_base):
    __tablename__ = 'datasets'
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True, nullable=False)
    description = sqla.Column(sqla.String)
    username = sqla.Column(sqla.String)
    creation_time = sqla.Column(sqla.DateTime(timezone=False), server_default=sqla.sql.func.now())
    size = sqla.Column(sqla.Integer)    # Number of datapoints in the dataset
    extra_info = sqla.Column(sqla.JSON)
    evaluation_settings = sqla.orm.relationship('EvaluationSetting', back_populates='dataset', cascade='all, delete, delete-orphan', foreign_keys='EvaluationSetting.dataset_uuid')
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<Dataset(uuid="{self.uuid}", name="{self.name}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class EvaluationSetting(sqlalchemy_base):
    __tablename__ = 'evaluation_settings'
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True, nullable=False)
    description = sqla.Column(sqla.String)
    username = sqla.Column(sqla.String)
    creation_time = sqla.Column(sqla.DateTime(timezone=False), server_default=sqla.sql.func.now())
    dataset_uuid = sqla.Column(sqla.String, sqla.ForeignKey('datasets.uuid'), nullable=False)
    dataset = sqla.orm.relationship('Dataset', back_populates='evaluation_settings', foreign_keys=[dataset_uuid])
    evaluations = sqla.orm.relationship('Evaluation', back_populates='setting', cascade='all, delete, delete-orphan', foreign_keys='Evaluation.setting_uuid')
    raw_inputs = sqla.orm.relationship('RawInput', back_populates='setting', cascade='all, delete, delete-orphan', foreign_keys='RawInput.setting_uuid')
    extra_info = sqla.Column(sqla.JSON)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<EvaluationSetting(uuid="{self.uuid}", name="{self.name}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


# For the raw float32 inputs we have for external models
class RawInput(sqlalchemy_base):
    __tablename__ = 'raw_inputs'
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True)
    description = sqla.Column(sqla.String)
    username = sqla.Column(sqla.String)
    creation_time = sqla.Column(sqla.DateTime(timezone=False), server_default=sqla.sql.func.now())
    setting_uuid = sqla.Column(sqla.String, sqla.ForeignKey('evaluation_settings.uuid'), nullable=False)
    setting = sqla.orm.relationship('EvaluationSetting', back_populates='raw_inputs', foreign_keys=[setting_uuid])
    data_shape = sqla.Column(sqla.JSON)
    data_format = sqla.Column(sqla.String)   # numpy type
    evaluations = sqla.orm.relationship('Evaluation', back_populates='raw_input', cascade='all, delete, delete-orphan', foreign_keys='Evaluation.raw_input_uuid')
    extra_info = sqla.Column(sqla.JSON)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<RawInput(uuid="{self.uuid}", name="{self.name}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class Evaluation(sqlalchemy_base):
    __tablename__ = 'evaluations'
    uuid = sqla.Column(sqla.String, primary_key=True)
    name = sqla.Column(sqla.String, unique=True)
    creation_time = sqla.Column(sqla.DateTime(timezone=False), server_default=sqla.sql.func.now())
    checkpoint_uuid = sqla.Column(sqla.String, sqla.ForeignKey('checkpoints.uuid'), nullable=False)
    checkpoint = sqla.orm.relationship('Checkpoint', back_populates='evaluations', foreign_keys=[checkpoint_uuid])
    # TODO: eventually make this nullable=False
    setting_uuid = sqla.Column(sqla.String, sqla.ForeignKey('evaluation_settings.uuid'), nullable=True)
    setting = sqla.orm.relationship('EvaluationSetting', back_populates='evaluations', foreign_keys=[setting_uuid])
    raw_input_uuid = sqla.Column(sqla.String, sqla.ForeignKey('raw_inputs.uuid'), nullable=True)
    raw_input = sqla.orm.relationship('RawInput', back_populates='evaluations', foreign_keys=[raw_input_uuid])
    chunks = sqla.orm.relationship('EvaluationChunk', back_populates='evaluation', cascade='all, delete, delete-orphan', foreign_keys='EvaluationChunk.evaluation_uuid')
    username = sqla.Column(sqla.String)
    extra_info = sqla.Column(sqla.JSON)
    completed = sqla.Column(sqla.Boolean)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<Evaluation(uuid="{self.uuid}", checkpoint_uuid="{self.checkpoint_uuid}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.__hash__() == hash(other)


class EvaluationChunk(sqlalchemy_base):
    __tablename__ = 'evaluation_chunks'
    uuid = sqla.Column(sqla.String, primary_key=True)
    creation_time = sqla.Column(sqla.DateTime(timezone=False), server_default=sqla.sql.func.now())
    evaluation_uuid = sqla.Column(sqla.String, sqla.ForeignKey('evaluations.uuid'), nullable=False)
    evaluation = sqla.orm.relationship('Evaluation', back_populates='chunks', foreign_keys=[evaluation_uuid])
    username = sqla.Column(sqla.String)
    extra_info = sqla.Column(sqla.JSON)
    hidden = sqla.Column(sqla.Boolean)

    def __repr__(self):
        return f'<EvaluationChunk(uuid="{self.uuid}", evaluation_uuid="{self.evaluation_uuid}")>'

    def __hash__(self):
        return hash(hash(self.uuid) + hash(self.name))

    def __eq__(self, other):
        return self.hash() == hash(other)


class ModelRepository:
    def __init__(self, mode=s3_utils.DB_CONNECTION_MODE, sql_verbose=False, download_database=True):
        self.sql_verbose = sql_verbose
        if mode == "sqlite":
            if download_database:
                download_db()
            self.db_connection_string = s3_utils.DB_CONNECTION_STRING_SQLITE
            self.engine = sqla.create_engine(self.db_connection_string, echo=self.sql_verbose)
        elif mode == "rds":
            self.db_connection_string = s3_utils.DB_CONNECTION_STRING_RDS
            self.engine = sqla.create_engine(self.db_connection_string, echo=self.sql_verbose, pool_pre_ping=True)
        else:
            assert False
        if not database_exists(self.engine.url):
            create_database(self.engine.url)
        self.sessionmaker = sqla.orm.sessionmaker(bind=self.engine, expire_on_commit=False)
        self.cache_root_path = s3_utils.default_cache_root_path
        self.s3wrapper = s3_utils.S3Wrapper(bucket='robustness-eval', cache_root_path=self.cache_root_path, verbose=False)
        self.uuid_length = 10
        self.pickle_protocol = 4

    def dispose(self):
        self.engine.dispose()

    @contextlib.contextmanager
    def session_scope(self):
        session = self.sessionmaker()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

    def gen_short_uuid(self):
        new_id = gen_short_uuid(num_chars=self.uuid_length)
        # TODO: check that we don't have a collision with the db?
        return new_id

    def gen_checkpoint_uuid(self):
        return gen_short_uuid(num_chars=None)

    def run_query_with_optional_session(self, query, session=None):
        if session is None:
            with self.session_scope() as sess:
                return query(sess)
        else:
            return query(session)

    def run_get(self, get_fn, session=None, assert_exists=True):
        def query(sess):
            result = get_fn(sess)
            assert len(result) <= 1
            if assert_exists:
                assert len(result) == 1
            if len(result) == 0:
                return None
            else:
                return result[0]
        return self.run_query_with_optional_session(query, session)

    def get_model(self, *,
                  uuid=None,
                  name=None,
                  session=None,
                  assert_exists=True,
                  load_final_checkpoint=False,
                  load_all_checkpoints=False,
                  load_evaluations=False):
        if uuid is not None:
            assert type(uuid) is str
        if name is not None:
            assert type(name) is str
        def get_fn(sess):
            return self.get_models(uuids=[uuid] if uuid is not None else None,
                                   names=[name] if name is not None else None,
                                   session=sess,
                                   load_final_checkpoint=load_final_checkpoint,
                                   load_all_checkpoints=load_all_checkpoints,
                                   load_evaluations=load_evaluations,
                                   show_hidden=True)
        return self.run_get(get_fn, session=session, assert_exists=assert_exists)

    def get_checkpoint(self, uuid=None, *, session=None, assert_exists=True, load_parents=False, load_evaluations=False):
        if uuid is not None:
            assert type(uuid) is str
        def get_fn(sess):
            return self.get_checkpoints(uuids=[uuid], session=sess, load_parents=load_parents, load_evaluations=load_evaluations, show_hidden=True)
        return self.run_get(get_fn, session=session, assert_exists=assert_exists)

    def get_dataset(self, *,
                    uuid=None,
                    name=None,
                    session=None,
                    assert_exists=True,
                    load_evaluation_settings=False):
        if uuid is not None:
            assert type(uuid) is str
        if name is not None:
            assert type(name) is str
        def get_fn(sess):
            return self.get_datasets(uuids=[uuid] if uuid is not None else None,
                                     names=[name] if name is not None else None,
                                     session=sess,
                                     load_evaluation_settings=load_evaluation_settings,
                                     show_hidden=True)
        return self.run_get(get_fn, session=session, assert_exists=assert_exists)

    def get_evaluation_setting(self, *,
                               uuid=None,
                               name=None,
                               session=None,
                               assert_exists=True,
                               load_parents=False,
                               load_evaluations=False,
                               load_raw_inputs=False):
        if uuid is not None:
            assert type(uuid) is str
        if name is not None:
            assert type(name) is str
        def get_fn(sess):
            return self.get_evaluation_settings(uuids=[uuid] if uuid is not None else None,
                                                names=[name] if name is not None else None,
                                                session=sess,
                                                load_parents=load_parents,
                                                load_evaluations=load_evaluations,
                                                load_raw_inputs=load_raw_inputs,
                                                show_hidden=True)
        return self.run_get(get_fn, session=session, assert_exists=assert_exists)

    def get_raw_input(self, uuid=None, *, session=None, assert_exists=True, load_parents=False, load_evaluations=False):
        if uuid is not None:
            assert type(uuid) is str
        def get_fn(sess):
            return self.get_raw_inputs(uuids=[uuid], session=sess, load_parents=load_parents, load_evaluations=load_evaluations,  show_hidden=True)
        return self.run_get(get_fn, session=session, assert_exists=assert_exists)

    def get_evaluation(self, uuid=None, *, session=None, assert_exists=True, load_parents=False, load_chunks=True):
        if uuid is not None:
            assert type(uuid) is str
        def get_fn(sess):
            return self.get_evaluations(uuids=[uuid], session=sess, load_parents=load_parents, load_chunks=load_chunks, show_hidden=True)
        return self.run_get(get_fn, session=session, assert_exists=assert_exists)

    def get_evaluation_chunk(self, *, uuid=None, session=None, assert_exists=True, load_parents=False):
        if uuid is not None:
            assert type(uuid) is str
        def get_fn(sess):
            return self.get_evaluation_chunks(uuids=[uuid], session=sess, load_parents=load_parents, show_hidden=True)
        return self.run_get(get_fn, session=session, assert_exists=assert_exists)

    def model_uuid_exists(self, uuid, session=None):
        return self.get_model(uuid=uuid, assert_exists=False, session=session) is not None

    def checkpoint_uuid_exists(self, uuid, session=None):
        return self.get_checkpoint(uuid=uuid, assert_exists=False, session=session) is not None

    def dataset_uuid_exists(self, uuid, session=None):
        return self.get_dataset(uuid=uuid, assert_exists=False, session=session) is not None

    def evaluation_setting_uuid_exists(self, uuid, session=None):
        return self.get_evaluation_setting(uuid=uuid, assert_exists=False, session=session) is not None

    def raw_input_uuid_exists(self, uuid, session=None):
        return self.get_raw_input(uuid=uuid, assert_exists=False, session=session) is not None

    def evaluation_uuid_exists(self, uuid, session=None):
        return self.get_evaluation(uuid=uuid, assert_exists=False, session=session) is not None

    def evaluation_chunk_uuid_exists(self, uuid, session=None):
        return self.get_evaluation_chunk(uuid=uuid, assert_exists=False, session=session) is not None

    def get_checkpoints(self, uuids=None, *, session=None, load_parents=True, load_evaluations=False, show_hidden=False):
        cur_options = []
        if load_parents:
            cur_options.append(sqla.orm.subqueryload(Checkpoint.model))
        if load_evaluations:
            cur_options.append(sqla.orm.subqueryload(Checkpoint.evaluations))
        filter_list = []
        if not show_hidden:
            filter_list.append(Checkpoint.hidden == False)
        if uuids is not None:
            filter_list.append(Checkpoint.uuid.in_(uuids))
        def query(sess):
            return sess.query(Checkpoint).options(cur_options).filter(*filter_list).all()
        return self.run_query_with_optional_session(query, session)

    def get_datasets(self, *,
                     uuids=None,
                     names=None,
                     session=None,
                     load_evaluation_settings=True,
                     show_hidden=False):
        cur_options = []
        if load_evaluation_settings:
            cur_options.append(sqla.orm.subqueryload(Dataset.evaluation_settings))
        filter_list = []
        if not show_hidden:
            filter_list.append(Dataset.hidden == False)
        if uuids is not None:
            filter_list.append(Dataset.uuid.in_(uuids))
        if names is not None:
            filter_list.append(Dataset.name.in_(names))
        def query(sess):
            return sess.query(Dataset).options(cur_options).filter(*filter_list).all()
        return self.run_query_with_optional_session(query, session)

    def get_evaluation_settings(self, *,
                                uuids=None,
                                names=None,
                                session=None,
                                load_parents=True,
                                load_evaluations=False,
                                load_raw_inputs=False,
                                show_hidden=False):
        cur_options = []
        if load_parents:
            cur_options.append(sqla.orm.subqueryload(EvaluationSetting.dataset))
        if load_evaluations:
            cur_options.append(sqla.orm.subqueryload(EvaluationSetting.evaluations).subqueryload(Evaluation.checkpoint).subqueryload(Checkpoint.model))
        if load_raw_inputs:
            cur_options.append(sqla.orm.subqueryload(EvaluationSetting.raw_inputs))
        filter_list = []
        if not show_hidden:
            filter_list.append(EvaluationSetting.hidden == False)
        if uuids is not None:
            filter_list.append(EvaluationSetting.uuid.in_(uuids))
        if names is not None:
            filter_list.append(EvaluationSetting.name.in_(names))
        def query(sess):
            return sess.query(EvaluationSetting).options(cur_options).filter(*filter_list).all()
        return self.run_query_with_optional_session(query, session)

    def get_raw_inputs(self, uuids=None, *, session=None, load_parents=True, load_evaluations=False, show_hidden=False):
        cur_options = []
        if load_parents:
            cur_options.append(sqla.orm.subqueryload(RawInput.setting).subqueryload(EvaluationSetting.dataset))
        if load_evaluations:
            cur_options.append(sqla.orm.subqueryload(RawInput.evaluations).subqueryload(Evaluation.checkpoint).subqueryload(Checkpoint.model))
        filter_list = []
        if not show_hidden:
            filter_list.append(RawInput.hidden == False)
        if uuids is not None:
            filter_list.append(RawInput.uuid.in_(uuids))
        def query(sess):
            return sess.query(RawInput).options(cur_options).filter(*filter_list).all()
        return self.run_query_with_optional_session(query, session)

    def get_evaluations(self, uuids=None, *, session=None, load_parents=True, load_chunks=True, show_hidden=False):
        cur_options = []
        if load_parents:
            cur_options.append(sqla.orm.subqueryload(Evaluation.checkpoint).subqueryload(Checkpoint.model))
            cur_options.append(sqla.orm.subqueryload(Evaluation.raw_input))
            cur_options.append(sqla.orm.subqueryload(Evaluation.setting).subqueryload(EvaluationSetting.dataset))
        if load_chunks:
            cur_options.append(sqla.orm.subqueryload(Evaluation.chunks))
        filter_list = []
        if not show_hidden:
            filter_list.append(Evaluation.hidden == False)
        if uuids is not None:
            filter_list.append(Evaluation.uuid.in_(uuids))
        def query(sess):
            return sess.query(Evaluation).options(cur_options).filter(*filter_list).all()
        return self.run_query_with_optional_session(query, session)

    def get_evaluation_chunks(self, *, uuids=None, session=None, load_parents=False, show_hidden=False):
        cur_options = []
        if load_parents:
            cur_options.append(sqla.orm.subqueryload(EvaluationChunk.evaluation))
        filter_list = []
        if not show_hidden:
            filter_list.append(EvaluationChunk.hidden == False)
        if uuids is not None:
            filter_list.append(EvaluationChunk.uuid.in_(uuids))
        def query(sess):
            return sess.query(EvaluationChunk).options(cur_options).filter(*filter_list).all()
        return self.run_query_with_optional_session(query, session)

    def get_models(self, *,
                   uuids=None,
                   names=None,
                   session=None,
                   load_parents=True,
                   load_final_checkpoint=True,
                   load_all_checkpoints=False,
                   load_evaluations=False,
                   show_hidden=False):
        cur_options = []
        checkpoint_nodes = []
        if load_final_checkpoint:
            cur_options.append(sqla.orm.subqueryload(Model.final_checkpoint))
            checkpoint_nodes.append(cur_options[-1])
        if load_all_checkpoints:
            cur_options.append(sqla.orm.subqueryload(Model.checkpoints))
            checkpoint_nodes.append(cur_options[-1])
        if load_evaluations:
            for opt in checkpoint_nodes:
                opt.subqueryload(Checkpoint.evaluations)
        filter_list = []
        if not show_hidden:
            filter_list.append(Model.hidden == False)
        if uuids is not None:
            filter_list.append(Model.uuid.in_(uuids))
        if names is not None:
            filter_list.append(Model.name.in_(names))
        def query(sess):
            return sess.query(Model).options(cur_options).filter(*filter_list).all()
        return self.run_query_with_optional_session(query, session)

    def create_model(self, extra_info=None, name=None, description=None, verbose=False, completed=False):
        with self.session_scope() as session:
            new_id = self.gen_short_uuid()
            username = getpass.getuser()
            new_model = Model(uuid=new_id,
                              name=name,
                              description=description,
                              username=username,
                              extra_info=extra_info,
                              hidden=False,
                              completed=completed,
                              logdir_filepaths={},
                              final_checkpoint_uuid=None)
            session.add(new_model)
        return self.get_model(uuid=new_id, assert_exists=True)

    def rename_model(self, model_uuid, new_name):
        with self.session_scope() as session:
            model = self.get_model(uuid=model_uuid, session=session, assert_exists=True)
            old_name = model.name
            model.name = new_name
        return old_name

    def hide_model(self, model_uuid):
        with self.session_scope() as session:
            model = self.get_model(uuid=model_uuid, session=session, assert_exists=True)
            model.hidden = True

    def get_latest_model_checkpoint_data(self, model_uuid, verbose=False, allow_non_final_checkpoint=True):
        with self.session_scope() as session:
            model = self.get_model(uuid=model_uuid, session=session, assert_exists=True)
            if len(model.checkpoints) == 0:
                return None, None
            if allow_non_final_checkpoint:
                cur_checkpoints = sorted(model.checkpoints, key=lambda x: x.training_step)
                checkpoint_to_load = cur_checkpoints[-1]
            else:
                assert model.final_checkpoint is not None
                checkpoint_to_load = model.final_checkpoint
            checkpoint_uuid = checkpoint_to_load.uuid
            key = get_checkpoint_data_key(checkpoint_uuid)
            if self.s3wrapper.exists(key):
                data = self.s3wrapper.get(key, verbose=verbose)
            else:
                data = None
            return data, checkpoint_to_load

    def mark_model_as_completed(self, model_uuid):
        with self.session_scope() as session:
            model = self.get_model(uuid=model_uuid, session=session, assert_exists=True)
            model.completed = True

    def set_final_model_checkpoint(self, model_uuid, checkpoint_uuid):
        with self.session_scope() as session:
            model = self.get_model(uuid=model_uuid, session=session, assert_exists=True)
            checkpoint = self.get_checkpoint(checkpoint_uuid, session=session, assert_exists=True)
            assert checkpoint.model_uuid == model_uuid
            model.final_checkpoint_uuid = checkpoint_uuid

    def store_logdir(self, model_uuid, logdir, verbose=False):
        with self.session_scope() as session:
            model = self.get_model(uuid=model_uuid, session=session, assert_exists=True)
            logdir_path = pathlib.Path(logdir).resolve()
            assert logdir_path.is_dir()
            tmp_filepaths = [x for x in logdir_path.glob('**/*') if x.is_file()]
            all_data = {}
            base_key = get_logdir_key(model_uuid) + '/'
            cur_logdir_files = {}
            for cur_filepath in tmp_filepaths:
                cur_filepath_resolved= cur_filepath.resolve()
                with open(cur_filepath_resolved, 'rb') as f:
                    cur_data = f.read()
                cur_relative_path = str(cur_filepath.relative_to(logdir_path))
                assert cur_relative_path not in cur_logdir_files
                cur_logdir_files[cur_relative_path] = {
                        'size': cur_filepath_resolved.stat().st_size,
                        'mtime': cur_filepath_resolved.stat().st_mtime}
                cur_key = base_key + cur_relative_path
                all_data[cur_key] = cur_data
            self.s3wrapper.put_multiple(all_data, verbose=verbose)
            model.logdir_filepaths = cur_logdir_files
            sqla.orm.attributes.flag_modified(model, 'logdir_filepaths')

    def create_checkpoint(self, *, model_uuid,
                          training_step=None,
                          epoch=None,
                          name=None,
                          data_bytes=None,
                          extra_info=None,
                          verbose=False):
        with self.session_scope() as session:
            assert self.model_uuid_exists(model_uuid, session=session)
            new_id = self.gen_checkpoint_uuid()
            username = getpass.getuser()
            new_checkpoint = Checkpoint(uuid=new_id,
                                        model_uuid=model_uuid,
                                        username=username,
                                        extra_info=extra_info,
                                        name=name,
                                        training_step=training_step,
                                        epoch=epoch,
                                        hidden=False)
            if data_bytes is not None:
                key = get_checkpoint_data_key(new_id)
                self.s3wrapper.put(data_bytes, key, verbose=verbose)
            session.add(new_checkpoint)
        return self.get_checkpoint(uuid=new_id, assert_exists=True)

    def get_checkpoint_data(self, checkpoint_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.checkpoint_uuid_exists(checkpoint_uuid, session=session)
            key = get_checkpoint_data_key(checkpoint_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def create_evaluation(self, *, checkpoint_uuid,
                          setting_uuid,
                          name=None,
                          logits_data_bytes=None,
                          extra_data_bytes=None,
                          raw_input_uuid=None,
                          extra_info=None,
                          completed=False,
                          verbose=False):
        with self.session_scope() as session:
            assert self.checkpoint_uuid_exists(checkpoint_uuid, session=session)
            assert self.evaluation_setting_uuid_exists(setting_uuid, session=session)
            if raw_input_uuid is not None:
                assert self.raw_input_uuid_exists(raw_input_uuid, session=session)
            new_id = self.gen_short_uuid()
            username = getpass.getuser()
            new_evaluation = Evaluation(uuid=new_id,
                                        checkpoint_uuid=checkpoint_uuid,
                                        setting_uuid=setting_uuid,
                                        raw_input_uuid=raw_input_uuid,
                                        username=username,
                                        extra_info=extra_info,
                                        name=name,
                                        completed=completed,
                                        hidden=False)
            if extra_data_bytes is not None:
                key = get_evaluation_extra_data_key(new_id)
                self.s3wrapper.put(extra_data_bytes, key, verbose=verbose)
            if logits_data_bytes is not None:
                key = get_evaluation_logits_data_key(new_id)
                self.s3wrapper.put(logits_data_bytes, key, verbose=verbose)
            session.add(new_evaluation)
        return self.get_evaluation(uuid=new_id, assert_exists=True)

    def hide_evaluation(self, evaluation_uuid):
        with self.session_scope() as session:
            evaluation = self.get_evaluation(evaluation_uuid, session=session, assert_exists=True)
            evaluation.hidden = True

    def rename_evaluation(self, evaluation_uuid, new_name):
        with self.session_scope() as session:
            evaluation = self.get_evaluation(evaluation_uuid, session=session, assert_exists=True)
            old_name = evaluation.name
            evaluation.name = new_name
        return old_name

    def mark_evaluation_as_completed(self, evaluation_uuid):
        with self.session_scope() as session:
            evaluation = self.get_evaluation(uuid=evaluation_uuid, session=session, assert_exists=True)
            evaluation.completed = True

    def get_evaluation_extra_data(self, evaluation_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_uuid_exists(evaluation_uuid, session=session)
            key = get_evaluation_extra_data_key(evaluation_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def get_evaluation_logits_data(self, evaluation_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_uuid_exists(evaluation_uuid, session=session)
            key = get_evaluation_logits_data_key(evaluation_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def has_evaluation_logits_data(self, evaluation_uuid):
        with self.session_scope() as session:
            assert self.evaluation_uuid_exists(evaluation_uuid, session=session)
            key = get_evaluation_logits_data_key(evaluation_uuid)
            return self.s3wrapper.exists(key)

    def put_evaluation_extra_data(self, evaluation_uuid, extra_data_bytes, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_uuid_exists(evaluation_uuid, session=session)
            key = get_evaluation_extra_data_key(evaluation_uuid)
            self.s3wrapper.put(extra_data_bytes, key, verbose=verbose)

    def put_evaluation_logits_data(self, evaluation_uuid, logits_data_bytes, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_uuid_exists(evaluation_uuid, session=session)
            key = get_evaluation_logits_data_key(evaluation_uuid)
            self.s3wrapper.put(logits_data_bytes, key, verbose=verbose)

    def create_dataset(self, *,
                       name,
                       size,
                       description=None,
                       data_bytes=None,
                       data_filename=None,    # use one of the two - directly uploading from a file can save memory
                       extra_info=None,
                       verbose=False):
        assert name is not None
        assert size is not None
        assert type(size) is int
        assert data_bytes is None or data_filename is None
        assert data_bytes is not None or data_filename is not None
        with self.session_scope() as session:
            new_id = self.gen_short_uuid()
            username = getpass.getuser()
            new_dataset = Dataset(uuid=new_id,
                                  name=name,
                                  description=description,
                                  username=username,
                                  size=size,
                                  extra_info=extra_info,
                                  hidden=False)
            key = get_dataset_data_key(new_id)
            if data_bytes is not None:
                self.s3wrapper.put(data_bytes, key, verbose=verbose)
            else:
                assert data_filename is not None
                self.s3wrapper.upload_file(data_filename, key, verbose=verbose)
            session.add(new_dataset)
        return self.get_dataset(uuid=new_id, assert_exists=True)

    def get_dataset_data(self, dataset_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.dataset_uuid_exists(dataset_uuid, session=session)
            key = get_dataset_data_key(dataset_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def download_dataset_data(self, dataset_uuid, target_filename, verbose=False):
        with self.session_scope() as session:
            assert self.dataset_uuid_exists(dataset_uuid, session=session)
            key = get_dataset_data_key(dataset_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.download_file(key, target_filename, verbose=verbose)

    def rename_dataset(self, dataset_uuid, new_name):
        with self.session_scope() as session:
            dataset = self.get_dataset(uuid=dataset_uuid, session=session, assert_exists=True)
            old_name = dataset.name
            dataset.name = new_name
        return old_name

    def hide_dataset(self, dataset_uuid):
        with self.session_scope() as session:
            dataset = self.get_dataset(uuid=dataset_uuid, session=session, assert_exists=True)
            dataset.hidden = True

    def create_evaluation_setting(self, *,
                                  name,
                                  dataset_uuid=None,
                                  description=None,
                                  extra_info=None,
                                  # in case the evaluation settings have a differently processe dataset associated with them
                                  processed_dataset_bytes=None,
                                  # use one of the two - directly uploading from a file can save memory
                                  processed_dataset_filename=None,
                                  extra_data_bytes=None,
                                  verbose=False):
        assert name is not None
        assert processed_dataset_filename is None or processed_dataset_bytes is None
        with self.session_scope() as session:
            if dataset_uuid is not None:
                assert self.dataset_uuid_exists(dataset_uuid, session=session)
            new_id = self.gen_short_uuid()
            username = getpass.getuser()
            new_setting = EvaluationSetting(uuid=new_id,
                                            name=name,
                                            description=description,
                                            username=username,
                                            dataset_uuid=dataset_uuid,
                                            extra_info=extra_info,
                                            hidden=False)
            if extra_data_bytes is not None:
                key = get_evaluation_setting_extra_data_key(new_id)
                self.s3wrapper.put(extra_data_bytes, key, verbose=verbose)
            key = get_evaluation_setting_processed_dataset_key(new_id)
            if processed_dataset_bytes is not None:
                self.s3wrapper.put(processed_dataset_bytes, key, verbose=verbose)
            elif processed_dataset_filename is not None:
                self.s3wrapper.upload_file(processed_dataset_filename, key, verbose=verbose)
            session.add(new_setting)
        return self.get_evaluation_setting(uuid=new_id, assert_exists=True)

    def hide_evaluation_setting(self, evaluation_setting_uuid):
        with self.session_scope() as session:
            evaluation_setting = self.get_evaluation_setting(uuid=evaluation_setting_uuid, session=session, assert_exists=True)
            evaluation_setting.hidden = True

    def rename_evaluation_setting(self, evaluation_setting_uuid, new_name):
        with self.session_scope() as session:
            evaluation_setting = self.get_evaluation_setting(uuid=evaluation_setting_uuid, session=session, assert_exists=True)
            old_name = evaluation_setting.name
            evaluation_setting.name = new_name
        return old_name

    def get_evaluation_setting_extra_data(self, evaluation_setting_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_setting_uuid_exists(evaluation_setting_uuid, session=session)
            key = get_evaluation_setting_extra_data_key(evaluation_setting_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def download_evaluation_setting_processed_dataset_data(self, evaluation_setting_uuid, target_filename, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_setting_uuid_exists(evaluation_setting_uuid, session=session)
            key = get_evaluation_setting_processed_dataset_key(evaluation_setting_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.download_file(key, target_filename, verbose=verbose)

    def get_evaluation_setting_processed_dataset_data(self, evaluation_setting_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_setting_uuid_exists(evaluation_setting_uuid, session=session)
            key = get_evaluation_setting_processed_dataset_key(evaluation_setting_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def create_raw_input(self, *,
                         name,
                         evaluation_setting_uuid,
                         data_shape,
                         data_format,
                         description=None,
                         extra_info=None,
                         data_bytes=None,
                         data_filename=None,   # use one of the two - directly uploading from a file can save memory
                         verbose=False):
        assert name is not None
        assert evaluation_setting_uuid is not None
        assert data_bytes is None or data_filename is None
        assert data_bytes is not None or data_filename is not None
        assert data_format in ['float32', 'float64', 'uint8']   # add more here if necessary
        assert type(data_shape) is list
        for x in data_shape:
            assert type(x) is int
        with self.session_scope() as session:
            assert self.evaluation_setting_uuid_exists(evaluation_setting_uuid, session=session)
            new_id = self.gen_short_uuid()
            username = getpass.getuser()
            new_raw_input = RawInput(uuid=new_id,
                                     name=name,
                                     description=description,
                                     username=username,
                                     data_shape=data_shape,
                                     data_format=data_format,
                                     setting_uuid=evaluation_setting_uuid,
                                     extra_info=extra_info,
                                     hidden=False)
            key = get_raw_input_data_key(new_id)
            if data_bytes is not None:
                self.s3wrapper.put(data_bytes, key, verbose=verbose)
            else:
                assert data_filename is not None
                self.s3wrapper.upload_file(data_filename, key, verbose=verbose)
            session.add(new_raw_input)
        return self.get_raw_input(uuid=new_id, assert_exists=True)

    def hide_raw_input(self, raw_input_uuid):
        with self.session_scope() as session:
            raw_input = self.get_raw_input(raw_input_uuid, session=session, assert_exists=True)
            raw_input.hidden = True

    def rename_raw_input(self, raw_input_uuid, new_name):
        with self.session_scope() as session:
            raw_input = self.get_raw_input(raw_input_uuid, session=session, assert_exists=True)
            old_name = raw_input.name
            raw_input.name = new_name
        return old_name

    def download_raw_input_data(self, raw_input_uuid, target_filename, verbose=False):
        with self.session_scope() as session:
            assert self.raw_input_uuid_exists(raw_input_uuid, session=session)
            key = get_raw_input_data_key(raw_input_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.download_file(key, target_filename, verbose=verbose)

    def get_raw_input_data(self, raw_input_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.raw_input_uuid_exists(raw_input_uuid, session=session)
            key = get_raw_input_data_key(raw_input_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def create_evaluation_chunk(self, *, evaluation_uuid,
                                         logits_data_bytes=None,
                                         indices=None,        # pass in either indices or indices_bytes
                                         indices_bytes=None,
                                         extra_data_bytes=None,
                                         extra_info=None,
                                         verbose=False):
        if indices is not None:
            assert indices_bytes is None
            indices_bytes = pickle.dumps(indices)
        else:
            assert indices_bytes is not None
        with self.session_scope() as session:
            assert self.evaluation_uuid_exists(evaluation_uuid, session=session)
            new_id = self.gen_short_uuid()
            username = getpass.getuser()
            new_chunk = EvaluationChunk(uuid=new_id,
                                        evaluation_uuid=evaluation_uuid,
                                        username=username,
                                        extra_info=extra_info,
                                        hidden=False)
            if extra_data_bytes is not None:
                key = get_evaluation_chunk_extra_data_key(new_id)
                self.s3wrapper.put(extra_data_bytes, key, verbose=verbose)
            if logits_data_bytes is not None:
                key = get_evaluation_chunk_logits_data_key(new_id)
                self.s3wrapper.put(logits_data_bytes, key, verbose=verbose)
            if indices_bytes is not None:
                key = get_evaluation_chunk_indices_data_key(new_id)
                self.s3wrapper.put(indices_bytes, key, verbose=verbose)
            session.add(new_chunk)
        return self.get_evaluation_chunk(uuid=new_id, assert_exists=True)

    def hide_evaluation_chunk(self, evaluation_chunk_uuid):
        with self.session_scope() as session:
            evaluation_chunk = self.get_evaluation_chunk(uuid=evaluation_chunk_uuid, session=session, assert_exists=True)
            evaluation_chunk.hidden = True

    def get_evaluation_chunk_extra_data(self, evaluation_chunk_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_chunk_uuid_exists(evaluation_chunk_uuid, session=session)
            key = get_evaluation_chunk_extra_data_key(evaluation_chunk_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def get_evaluation_chunk_logits_data(self, evaluation_chunk_uuid, verbose=False):
        with self.session_scope() as session:
            assert self.evaluation_chunk_uuid_exists(evaluation_chunk_uuid, session=session)
            key = get_evaluation_chunk_logits_data_key(evaluation_chunk_uuid)
            if self.s3wrapper.exists(key):
                return self.s3wrapper.get(key, verbose=verbose)
            else:
                return None

    def get_evaluation_chunk_indices_data(self, evaluation_chunk_uuid, verbose=False, unpickle=False):
        with self.session_scope() as session:
            assert self.evaluation_chunk_uuid_exists(evaluation_chunk_uuid, session=session)
            key = get_evaluation_chunk_indices_data_key(evaluation_chunk_uuid)
            if self.s3wrapper.exists(key):
                data = self.s3wrapper.get(key, verbose=verbose)
                if unpickle:
                    return pickle.loads(data)
                else:
                    return data
            else:
                return None
