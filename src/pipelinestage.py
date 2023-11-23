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

        # Initialize parameters
        self.params_dict = {}
        self.params = None
        self.raw_data_path = config.DATA_PATH_RAW

        try:
            with open(config.PARAMS_FILE_PATH, 'r') as file:
                self.params_dict = yaml.safe_load(file)
                self.params = Struct(self.params_dict)

            # If no name of data set is given, all files present in 'assets/data/raw'
            # will be used.
            if self.params.profile.dataset is not None:
                self.raw_data_path = config.DATA_PATH_RAW / self.params.profile.dataset

        except FileNotFoundError:
            print(f"Error: File {config.PARAMS_FILE_PATH} not found.")
            # Handle the error or re-raise as needed
        except yaml.YAMLError as exc:
            print(f"Error parsing the YAML file: {exc}")
            # Handle the error or re-raise as needed

    def save_data(self, dfs, filepaths):
        pass
    
    def load_model(self, model_filepath, method=None):

        if method is None:
            method = self.params.train.learning_method

        if method in config.DL_METHODS:
            model = models.load_model(model_filepath)
        else:
            model = load(model_filepath)

        return model
