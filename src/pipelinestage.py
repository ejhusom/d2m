#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class for pipeline stage.

Author:
    Erik Johannes Husom

Created:
    2023-08-02 onsdag 11:24:36 

"""
from pathlib import Path
import yaml
from pydantic import ValidationError
from joblib import load
from tensorflow.keras import models

from config import config
from param_validation import AllParams

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
    """
    A superclass for pipeline stages in a machine learning workflow.

    This class provides a common structure for individual stages of the 
    machine learning pipeline. It handles the initialization of parameters 
    from a configuration file, sets up the raw data path, and validates 
    the parameters using Pydantic models.

    Attributes:
        stage_name (str): The name of the current pipeline stage.
        params_dict (dict): A dictionary containing parameters from the YAML file.
        params (AllParams): Validated parameters using the AllParams Pydantic model.
        raw_data_path (Path): Path to the raw data directory, can be adjusted based on dataset parameter.

    Methods:
        __init__(stage_name): Constructs all the necessary attributes for the pipeline stage object.

    Args:
        stage_name (str): The name of the pipeline stage.

    Raises:
        FileNotFoundError: If the parameters YAML file is not found.
        yaml.YAMLError: If there are issues parsing the YAML file.
        ValidationError: If parameter validation fails.

    Example:
        >>> pipeline_stage = PipelineStage("clean")
        >>> print(pipeline_stage.stage_name)
        'clean'
    """

    def __init__(self, stage_name):

        self.stage_name = stage_name
        self.raw_data_path = config.DATA_PATH_RAW

        try:
            # Load and validate parameters
            with open(config.PARAMS_FILE_PATH, 'r') as file:
                params_dict = yaml.safe_load(file)
                self.params = AllParams(**params_dict)

            # Set the raw data path based on dataset parameter
            if self.params.profile.dataset:
                self.raw_data_path = Path(config.DATA_PATH_RAW) / self.params.profile.dataset

        except FileNotFoundError as e:
            print(f"Error: File {config.PARAMS_FILE_PATH} not found.")
            raise e
        except yaml.YAMLError as e:
            print(f"Error parsing the YAML file: {e}")
            raise e
        except ValidationError as e:
            print(f"Parameter validation error: {e}")
            raise e

    def save_data(self, dfs, filepaths):
        pass
    
    def load_model(self, model_filepath, method=None):
        model = None

        if method is None:
            method = self.params.train.learning_method

        try:
            if method in config.DL_METHODS:
                model = models.load_model(model_filepath)
            else:
                model = load(model_filepath)

        except FileNotFoundError:
            print(f"Error: Model file {model_filepath} not found.")
            # Handle the error or re-raise as needed

        return model
