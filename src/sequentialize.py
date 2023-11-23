#!/usr/bin/env python3
"""Split data into sequences.

Prepare the data for input to a neural network. A sequence with a given history
size is extracted from the input data, and matched with the appropriate target
value(s).

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import os
import sys

import numpy as np
import pandas as pd
import yaml
from codecarbon import track_emissions

from config import config
from pipelinestage import PipelineStage
from preprocess_utils import flatten_sequentialized, split_sequences


class SequentializeStage(PipelineStage):
    """Make sequences out of tabular data."""

    
    def __init__(self):
        super().__init__(stage_name="sequentialize")

    @track_emissions(project_name="sequentialize")
    def run(self):
        filepaths = self.find_files(config.DATA_SCALED_PATH, file_extension=".npz")

        if self.params.clean.classification:
            target_size = 1
        else:
            target_size = self.params.sequentialize.target_size

        output_columns = np.array(pd.read_csv(config.OUTPUT_FEATURES_PATH, index_col=0)).reshape(
            -1
        )

        n_output_cols = len(output_columns)

        for filepath in filepaths:

            infile = np.load(filepath)

            X = infile["X"]
            y = infile["y"]

            # Combine y and X to get correct format for sequentializing
            data = np.hstack((y, X))

            # Split into sequences
            X, y = split_sequences(
                data,
                self.params.sequentialize.window_size,
                target_size=target_size,
                n_target_columns=n_output_cols,
                future_predict=self.params.sequentialize.future_predict,
                overlap=self.params.sequentialize.overlap,
            )

            if self.params.train.learning_method in config.NON_SEQUENCE_LEARNING_METHODS or self.params.train.ensemble == True:
                X = flatten_sequentialized(X)

            if self.params.sequentialize.shuffle_samples:
                permutation = np.random.permutation(X.shape[0])
                X = np.take(X, permutation, axis=0)
                y = np.take(y, permutation, axis=0)

            # Save X and y into a binary file
            np.savez(
                config.DATA_SEQUENTIALIZED_PATH
                / (os.path.basename(filepath).replace("scaled.npz", "sequentialized.npz")),
                X=X,
                y=y,
                allow_pickle=True
            )


def main():
    SequentializeStage().run()

if __name__ == "__main__":
    main()
