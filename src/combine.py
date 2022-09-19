#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combines workout files into one.

Author:   
    Erik Johannes Husom

Created:  
    2020-10-29

"""
import os
import sys

import numpy as np

from config import DATA_COMBINED_PATH
from preprocess_utils import find_files


def combine(dir_path):
    """Combine data from multiple input files into one dataset.

    Args:
        dir_path (str): Path to directory containing files.

    """

    filepaths = find_files(dir_path, file_extension=".npz")

    DATA_COMBINED_PATH.mkdir(parents=True, exist_ok=True)

    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []
    calibrate_inputs = []
    calibrate_outputs = []

    for filepath in filepaths:
        infile = np.load(filepath)

        if "train" in filepath:
            train_inputs.append(infile["X"])
            train_outputs.append(infile["y"])
        elif "test" in filepath:
            test_inputs.append(infile["X"])
            test_outputs.append(infile["y"])
        elif "calibrate" in filepath:
            calibrate_inputs.append(infile["X"])
            calibrate_outputs.append(infile["y"])

    X_train = np.concatenate(train_inputs)
    y_train = np.concatenate(train_outputs)
    X_test = np.concatenate(test_inputs)
    y_test = np.concatenate(test_outputs)

    if len(calibrate_inputs) > 0:
        X_calibrate = np.concatenate(calibrate_inputs)
        y_calibrate = np.concatenate(calibrate_outputs)

    np.savez(DATA_COMBINED_PATH / "train.npz", X=X_train, y=y_train)
    np.savez(DATA_COMBINED_PATH / "test.npz", X=X_test, y=y_test)

    if len(calibrate_inputs) > 0:
        np.savez(DATA_COMBINED_PATH / "calibrate.npz", X=X_calibrate, y=y_calibrate)


if __name__ == "__main__":

    np.random.seed(2020)

    combine(sys.argv[1])
