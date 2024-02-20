#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performing inference using virtual sensors created with Erdre.

Author:
    Erik Johannes Husom

Created:
    2021-10-29 Friday 09:13:14 

"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from clean import *
from combine import *
from config import config
from evaluate import *
from featurize import *
from profiling import *
from scale import *
from sequentialize import *
from split import *
from train import *


class VirtualSensor:
    def __init__(
        self,
        params_file=config.PARAMS_FILE_PATH,
        profile_file=config.PROFILE_JSON_PATH,
        input_features_file=config.INPUT_FEATURES_PATH,
        output_features_file=config.OUTPUT_FEATURES_PATH,
        input_scaler_file=config.INPUT_SCALER_PATH,
        output_scaler_file=config.OUTPUT_SCALER_PATH,
        model_file=config.MODELS_FILE_PATH,
        verbose=True,
    ):
        """Initialize virtual sensor with the needed assets.

        To run a virtual sensor, certain assets are needed:

        - params.yaml: Parameters used to create virtual sensor.
        - profile.json: Profiling of training data set.
        - input_scaler.scl: Scaler used to transform input data.
        - input_scaler.scl: Scaler used to transform output data.
        - model.h5: Model.

        """

        if type(params_file) == dict:
            yaml.dump(params_file, open("params.yaml", "w"), allow_unicode=True)
            self.params_file = "params.yaml"
        else:
            self.params_file = params_file

        self.profile_file = profile_file
        self.input_features_file = input_features_file
        self.output_features_file = output_features_file
        self.input_scaler_file = input_scaler_file
        self.output_scaler_file = output_scaler_file
        self.model_file = model_file

        self.verbose = verbose

        self.assets_files = [
            self.params_file,
            self.profile_file,
            self.input_features_file,
            self.output_features_file,
            self.input_scaler_file,
            self.output_scaler_file,
            self.model_file,
        ]

        self._check_assets_existence()

    def _check_assets_existence(self):
        """Check if the needed assets exists."""

        check_ok = True

        for path in self.assets_files:
            if not os.path.exists(path):
                print(f"File {path} not found.")
                check_ok = False

        assert check_ok, "Assets missing."

    def run_virtual_sensor(self, inference_df):
        """Run virtual sensor.

        Args:
            TODO: What input format here? Currently it takes a CSV-file.
            input_data ():

        """

        params = yaml.safe_load(open("params.yaml"))
        classification = params["clean"]["classification"]
        onehot_encode_target = params["clean"]["onehot_encode_target"]
        learning_method = params["train"]["learning_method"]
        input_method = params["scale"]["input_method"]
        output_method = params["scale"]["output_method"]
        window_size = params["sequentialize"]["window_size"]
        overlap = params["sequentialize"]["overlap"]

        df = CleanStage().run(inference_df=inference_df)
        df = FeaturizeStage().run(inference_df=df)

        self._check_features_existence(df)

        X = np.array(df)

        input_scaler = joblib.load(config.INPUT_SCALER_PATH)
        output_scaler = joblib.load(config.OUTPUT_SCALER_PATH)

        if input_method is not None:
            X = input_scaler.transform(X)

        X = split_X_sequences(X, window_size, overlap=overlap)

        if learning_method in NON_SEQUENCE_LEARNING_METHODS:
            X = flatten_sequentialized(X)

        if learning_method in NON_DL_METHODS:
            model = load(config.MODELS_FILE_PATH)
        else:
            model = models.load_model(config.MODELS_FILE_PATH)

        y_pred = model.predict(X)

        if onehot_encode_target:
            y_pred = np.argmax(y_pred, axis=-1)
        elif classification:
            y_pred = np.array((y_pred > 0.5), dtype=np.int)

        print(y_pred)
        # plt.figure()
        # plt.plot(y_pred)
        # plt.show()

        return y_pred

    def _check_features_existence(self, inference_df):
        """Check that the DataFrame passed for inference contains the necessary
        features required by the virtual sensor.

        Args:
            inference_df (DataFrame): DataFrame to perform inference on.

        """

        input_features = pd.read_csv(self.input_features_file, index_col=0)
        print(input_features)
        print(inference_df.columns)

        for feature in input_features["0"]:
            assert (
                feature in inference_df.columns
            ), f"Input data does not contain {feature}, which is required to run virtual sensor."


if __name__ == "__main__":

    # df = pd.read_csv("./assets/data/raw/cnc_without_target/02.csv")
    df = pd.read_csv("./assets/data/raw/cnc_milling/02.csv")

    vs = VirtualSensor()
    vs.run_virtual_sensor(inference_df=df)
