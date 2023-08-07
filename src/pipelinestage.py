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

from config import *

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
        self.params = Struct(yaml.safe_load(open(PARAMS_FILE_PATH)))
    
    def load_model(self, model_filepath):

        if self.params.train.learning_method in NON_DL_METHODS:
            self.model = load(model_filepath)
        else:
            self.model = models.load_model(model_filepath)

        return model
