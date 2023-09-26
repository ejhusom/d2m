#!/usr/bin/env python3
"""Scaling the inputs of the data set.

Possible scaling methods

TODO:
    Implement scaling when there is only one workout file.

Author:
    Erik Johannes Husom

Created:
    2020-09-16

"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import yaml
from codecarbon import track_emissions
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import config
from preprocess_utils import find_files
from pipelinestage import PipelineStage

@track_emissions(project_name="scale")
class ScaleStage(PipelineStage):
    def __init__(self):
        super().__init__(stage_name="scale")

        # Initialize the scalers using the method
        self.input_scaler = self._get_scaler(self.params.scale.input)
        self.output_scaler = self._get_scaler(self.params.scale.output)

        if self.input_scaler == NotImplemented:
            raise NotImplementedError(f"{self.params['input']} not implemented.")
        if self.output_scaler == NotImplemented:
            raise NotImplementedError(f"{self.params['output']} not implemented.")


    def run(self):

        filepaths = find_files(config.DATA_SPLIT_PATH, file_extension=".npy")
        train_inputs = []
        train_outputs = []

        data_overview = {}

        output_columns = np.array(pd.read_csv(config.OUTPUT_FEATURES_PATH, index_col=0)).reshape(
            -1
        )

        n_output_cols = len(output_columns)

        for filepath in filepaths:

            data = np.load(filepath, allow_pickle=True)

            # Split into input (X) and output/target (y)
            X = data[:, n_output_cols:].copy()
            y = data[:, 0:n_output_cols].copy()

            # If we have a one-hot encoding of categorical labels, shape of y stays
            # the same, otherwise it is reshaped.
            # TODO: Make a better test
            # if classification and len(np.unique(y, axis=-1)) > 2:
            #     pass
            # else:
            if not self.params.clean.onehot_encode_target:
                y = y.reshape(-1, 1)

            if "train" in filepath:
                train_inputs.append(X)
                train_outputs.append(y)
                category = "train"
            elif "test" in filepath:
                category = "test"
            elif "calibrate" in filepath:
                category = "calibrate"

            data_overview[filepath] = {"X": X, "y": y, "category": category}

        X_train = np.concatenate(train_inputs)
        y_train = np.concatenate(train_outputs)

        # Fit a scaler to the training data
        self.input_scaler = self.input_scaler.fit(X_train)

        if not self.params.clean.classification:
            self.output_scaler = self.output_scaler.fit(y_train)

        for filepath in data_overview:

            # # Scale inputs
            # if input_method == None:
            #     X = data_overview[filepath]["X"]
            # else:
            X = self.input_scaler.transform(data_overview[filepath]["X"])

            # # Scale outputs
            # if output_method == None or self.params.clean.classification:
            #     y = data_overview[filepath]["y"]
            # else:
            y = self.output_scaler.transform(data_overview[filepath]["y"])

            # Save X and y into a binary file
            np.savez(
                config.DATA_SCALED_PATH
                / (
                    os.path.basename(filepath).replace(
                        data_overview[filepath]["category"] + ".npy",
                        data_overview[filepath]["category"] + "-scaled.npz",
                    )
                ),
                X=X,
                y=y,
            )

            joblib.dump(self.input_scaler, config.INPUT_SCALER_PATH)
            joblib.dump(self.output_scaler, config.OUTPUT_SCALER_PATH)

    @staticmethod
    def _get_scaler(method):
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            None: NoScaler()
        }
        return scalers.get(method, NotImplemented)


class NoScaler:
    def fit(self, data):
        return self
    
    def transform(self, data):
        return data

    def fit_transform(self, data):
        return data

def main():
    ScaleStage().run()

if __name__ == "__main__":
    main()
