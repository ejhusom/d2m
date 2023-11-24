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
from preprocess_utils import flatten_sequentialized

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

def split_sequences(
    sequences,
    window_size,
    target_size=1,
    n_target_columns=1,
    future_predict=False,
    overlap=0,
):
    """Split data sequence into samples with matching input and targets.

    Args:
        sequences (array): The matrix containing the sequences, with the
            targets in the first columns.
        window_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        target_size (int): Size of target window. Default=1, i.e. only one
            value is used as target.
        n_target_columns: Number of target columns. Default=1.
        future_predict (bool): Whether to predict target values backwards or
            forward from the last time step in the input sequence.
            Default=False, which means that the number of target values will be
            counted backwards.
        overlap (int): How many time steps to overlap for each sequence. If
            overlap is greater than window_size, it will be set to
            window_size-1, which is the largest overlap possible.  Default=0,
            which means there will be no overlap.

    Returns:
        X (array): The input samples.
        y (array): The targets.

    """
    X, y = list(), list()

    start_idx = 0

    # overlap can maximum be one less than window_size
    if overlap >= window_size:
        overlap = window_size - 1

    # for i in range(len(sequences)):
    while start_idx + window_size <= len(sequences):

        # find the end of this pattern
        end_ix = start_idx + window_size

        # find start of target window
        if future_predict:
            target_start_ix = end_ix
            target_end_ix = end_ix + target_size
        else:
            target_start_ix = end_ix - target_size
            target_end_ix = end_ix

        # check if we are beyond the dataset
        # if end_ix > len(sequences):
        if target_end_ix > len(sequences):
            break

        # Select all cols from sequences except target col, which leaves inputs
        seq_x = sequences[start_idx:end_ix, n_target_columns:]

        # Extract targets from sequences
        if n_target_columns > 1:
            seq_y = sequences[target_start_ix:target_end_ix, 0:n_target_columns]
            seq_y = seq_y.reshape(-1)
        else:
            seq_y = sequences[target_start_ix:target_end_ix, 0]

        # Skip round if target_size is not correct. May happen if target_size
        # is larger than window_size.
        # if len(seq_y) != target_size:
        #     start_idx += window_size - overlap
        #     continue

        X.append(seq_x)
        y.append(seq_y)

        start_idx += window_size - overlap

    X = np.array(X)
    y = np.array(y)

    return X, y

def split_X_sequences(sequences, window_size, overlap=0):
    """Split data sequence into samples with matching input and targets.

    Args:
        sequences (array): The matrix containing the sequences, with the
            targets in the first columns.
        window_size (int): Number of time steps to include in each sample, i.e.
            how much history should be matched with a given target.
        overlap (int): How many time steps to overlap for each sequence. If
            overlap is greater than window_size, it will be set to
            window_size-1, which is the largest overlap possible.  Default=0,
            which means there will be no overlap.

    Returns:
        X (array): Sequences from input array.

    """
    X = list()

    start_idx = 0

    if overlap >= window_size:
        overlap = window_size - 1

    while start_idx + window_size <= len(sequences):

        # find the end of this pattern
        end_ix = start_idx + window_size

        # Select all cols from sequences except target col, which leaves inputs
        seq_x = sequences[start_idx:end_ix, :]

        X.append(seq_x)

        start_idx += window_size - overlap

    X = np.array(X)

    return X

def flatten_sequentialized(X):
    """Flatten sequentialized data.

    Args:
        X (array): Array of shape [num_sequences, window_size, num_features].

    Returns:
        X_flat (array): Array of shape [num_sequences, window_size*num_features].

    """

    X_flat = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    return X_flat

def main():
    SequentializeStage().run()

if __name__ == "__main__":
    main()
