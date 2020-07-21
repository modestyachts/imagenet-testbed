from importlib import import_module
from pathlib import Path


class Registry():
    def __init__(self):
        self.models = {}
        self.eval_settings = {}

    def add_model(self, model):
        assert model.name not in self.models, \
               f'Duplicate model {model.name} found. Model names must be unique.'
        self.models[model.name] = model

    def add_eval_setting(self, eval_setting):
        assert eval_setting.name not in self.eval_settings, \
               f'Duplicate eval setting {eval_setting.name} found. Eval setting names must be unique.'
        self.eval_settings[eval_setting.name] = eval_setting

    def load_full_registry(self):
        for f in Path(__file__).parent.glob("models/*.py"):
            if '__' not in f.stem and str(f.stem) not in ['model_base']:
                import_module(f'models.{f.stem}')
                
        for f in Path(__file__).parent.glob("eval_settings/*.py"):
            if '__' not in f.stem and str(f.stem) not in ['eval_setting_base', 'eval_setting_subsample', 'image_utils']:
                import_module(f'eval_settings.{f.stem}')

    def model_names(self):
        return self.models.keys()

    def eval_setting_names(self):
        return self.eval_settings.keys()

    def contains_model(self, model_name):
        return model_name in self.models

    def contains_eval_setting(self, eval_setting_name):
        return eval_setting_name in self.eval_settings

    def get_model(self, model_name):
        return self.models[model_name]

    def get_eval_setting(self, eval_setting_name):
        return self.eval_settings[eval_setting_name]


registry = Registry()
