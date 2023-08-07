#!/usr/bin/env python3
"""Split data into training and test set.

Author:
    Erik Johannes Husom

Date:
    2021-02-24

"""
import os
import random
import shutil
import sys

import numpy as np
import yaml
from codecarbon import track_emissions

from config import config
from pipelinestage import PipelineStage
from preprocess_utils import find_files


class SplitStage(PipelineStage):
    """Split data into train and test set.

    Training files and test files are saved to different folders.

    Args:
        dir_path (str): Path to directory containing files.

    """

    def __init__(self):
        super().__init__(stage_name="clean")

    def run(self):

        filepaths = find_files(config.DATA_FEATURIZED_PATH, file_extension=".npy")

        # Handle special case where there is only one data file.
        if isinstance(filepaths, str) or len(filepaths) == 1:
            filepath = filepaths[0]

            data = np.load(filepath)

            if self.params.split.shuffle_samples_before_split:
                permutation = np.random.permutation(data.shape[0])
                data = np.take(data, permutation, axis=0)

            train_size = int(len(data) * self.params.split.train_split)

            data_train = None
            data_test = None

            data_train = data[:train_size, :]
            data_test = data[train_size:, :]

            np.save(
                config.DATA_SPLIT_PATH / os.path.basename(filepath).replace("featurized", "train"),
                data_train,
            )

            np.save(
                config.DATA_SPLIT_PATH / os.path.basename(filepath).replace("featurized", "test"),
                data_test,
            )

        else:

            if self.params.split.shuffle_files:
                random.shuffle(filepaths)

            # Parameter 'train_split' is used to find out no. of files in training set
            file_split = int(len(filepaths) * self.params.split.train_split)

            training_files = []
            test_files = []

            training_files = filepaths[:file_split]
            test_files = filepaths[file_split:]

            for filepath in filepaths:

                if filepath in training_files:
                    shutil.copyfile(
                        filepath,
                        config.DATA_SPLIT_PATH
                        / os.path.basename(filepath).replace("featurized", "train"),
                    )

                elif filepath in test_files:
                    shutil.copyfile(
                        filepath,
                        config.DATA_SPLIT_PATH
                        / os.path.basename(filepath).replace("featurized", "test"),
                    )

def main():
    SplitStage().run()

if __name__ == "__main__":
    main()
