#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class for pipeline stage.

Author:
    Erik Johannes Husom

Created:
    2023-08-02 onsdag 11:24:36 

"""
import yaml
from joblib import load
from tensorflow.keras import models

from config import config

class Struct(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value

class PipelineStage():

    def __init__(self, stage_name):

        self.stage_name = stage_name

        # Read parameter file and convert to object
        self.params_dict = yaml.safe_load(open(config.PARAMS_FILE_PATH))
        self.params = Struct(self.params_dict)

        self.raw_data_path = config.DATA_PATH_RAW

        # If no name of data set is given, all files present in 'assets/data/raw'
        # will be used.
        if self.params.profile.dataset is not None:
            self.raw_data_path = config.DATA_PATH_RAW / self.params.profile.dataset
    
    def load_model(self, model_filepath):

        if self.params.train.learning_method in config.NON_DL_METHODS:
            self.model = load(model_filepath)
        else:
            self.model = models.load_model(model_filepath)

        return model
