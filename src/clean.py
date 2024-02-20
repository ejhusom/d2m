#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean up data.

Author:
    Erik Johannes Husom

Created:
    2021-06-30

"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml
from codecarbon import track_emissions
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from config import config
from pipelinestage import PipelineStage

class CleanStage(PipelineStage):

    def __init__(self):
        super().__init__(stage_name="clean")

    @track_emissions(project_name="clean")
    def run(self, inference_df=None):

        if inference_df is None:
            removable_features = self.parse_profile_warnings()
            pd.DataFrame(removable_features).to_csv(config.REMOVABLE_FEATURES)
            filepaths = self.find_files(self.raw_data_path, file_extension=".csv")
            dfs = self.read_data(filepaths)
        else:
            removable_features = self.load_removable_features()
            dfs = [inference_df]

        dfs = self.remove_features(dfs, removable_features)
        combined_df = pd.concat(dfs, ignore_index=True)


        if inference_df is not None:
            # combined_df, output_columns = self.handle_target_encoding(combined_df)
            # self.save_data(combined_df, filepaths, output_columns)

            if self.params.clean.target in inference_df.columns:
                del combined_df[self.params.clean.target]

            return combined_df

        if self.params.clean.classification:

            if self.params.clean.onehot_encode_target and len(np.unique(combined_df[self.params.clean.target])) > 2:
                encoder = LabelBinarizer()
            else:
                if self.params.clean.onehot_encode_target:
                    raise ValueError(
                        "Parameter 'onehot_encode_target' is set to True, but target is binary. Change parameter to False in order to use this pipeline."
                    )
                encoder = LabelEncoder()

            target_col = np.array(combined_df[self.params.clean.target]).reshape(-1)
            encoder.fit(target_col)
            # print(f"Classes: {encoder.classes_}")
            # print(f"Encoded classes: {encoder.transform(encoder.classes_)}")

            combined_df, output_columns = self.encode_target(encoder, combined_df, self.params.clean.target)

            for i in range(len(dfs)):
                dfs[i], _ = self.encode_target(encoder, dfs[i], self.params.clean.target)

        else:
            output_columns = [self.params.clean.target]

        if self.params.clean.combine_files:
            combined_df.to_csv(config.DATA_CLEANED_PATH / (os.path.basename("data-cleaned.csv")))
        else:
            for filepath, df in zip(filepaths, dfs):
                df.to_csv(
                    config.DATA_CLEANED_PATH
                    / (os.path.basename(filepath).replace(".", "-cleaned."))
                )

        pd.DataFrame(output_columns).to_csv(config.OUTPUT_FEATURES_PATH)

    def load_removable_features(self):
        try:
            removable_features = np.array(
                pd.read_csv(config.REMOVABLE_FEATURES, index_col=0)
            ).reshape(-1)
            return removable_features
        except FileNotFoundError:
            print(f"Removable features file not found: {config.REMOVABLE_FEATURES}")
            raise

    def remove_features(self, dfs, removable_features):
        """Read data and delete removable features.

        Args:
            dfs (list of DataFrames): Data frames to read.
            removable_features (list): Features/columns to remove.

        Returns:
            cleaned_dfs (list of DataFrames): Data frames with removeable features
                removed.

        """

        cleaned_dfs = []

        for df in dfs:

            # If the first column is an index column, remove it.
            if df.iloc[:, 0].is_monotonic_increasing:
                df = df.iloc[:, 1:]

            for column in removable_features:
                if column in df:
                    del df[column]

            df.dropna(inplace=True)

            cleaned_dfs.append(df)

        return cleaned_dfs

    def handle_target_encoding(self, df):
        """
        Handle the encoding of the target variable in the dataframe.

        Args:
            df (pd.DataFrame): The dataframe to encode the target variable in.

        Returns:
            Tuple[pd.DataFrame, List[str]]: A tuple containing the modified dataframe and a list of output column names.
        """
        if self.params.clean.classification:
            if self.params.clean.onehot_encode_target and len(np.unique(df[self.params.clean.target])) > 2:
                encoder = LabelBinarizer()
            else:
                if self.params.clean.onehot_encode_target:
                    raise ValueError(
                        "Parameter 'onehot_encode_target' is set to True, but target is binary. Change parameter to False in order to use this pipeline."
                    )
                encoder = LabelEncoder()

            target_col = np.array(df[self.params.clean.target]).reshape(-1)
            encoder.fit(target_col)
            logging.info(f"Classes: {encoder.classes_}")
            logging.info(f"Encoded classes: {encoder.transform(encoder.classes_)}")

            df, output_columns = self.encode_target(encoder, df, self.params.clean.target)
            return df, output_columns
        else:
            return df, [self.params.clean.target]


    def encode_target(self, encoder, df, target):
        """Encode a target variable based on a fitted encoder.

        Args:
            encoder: A fitted encoder.
            df (DataFrame): DataFrame containing the target variable.
            target (str): Name of the target variable.

        Returns:
            df (DataFrame): DataFrame with the original target variable removed,
                substituted by a onehot encoding of the variable.
            output_columns (list): List of the names of the target columns.

        """

        output_columns = []

        target_col = np.array(df[target]).reshape(-1)
        target_encoded = encoder.transform(target_col)

        del df[target]

        if len(target_encoded.shape) > 1:
            for i in range(target_encoded.shape[-1]):
                column_name = f"{target}_{i}"
                df[column_name] = target_encoded[:, i]
                output_columns.append(column_name)
        else:
            df[target] = target_encoded
            output_columns.append(target)

        return df, output_columns


    def parse_profile_warnings(self):
        """Read profile warnings and find which columns to delete.

        Returns:
            removable_features (list): Which columns to delete from data set.

        """

        profile_json = json.load(open(config.PROFILE_JSON_PATH))

        # In some versions of pandas-profiling, 'messages' are called 'alerts'.
        try:
            messages = profile_json["messages"]
        except:
            messages = profile_json["alerts"]

        variables = list(profile_json["variables"].keys())

        try:
            correlations = profile_json["correlations"]["auto"]
        except:
            correlations = None

        removable_features = []

        for message in messages:
            # Find the names of variables
            matches = re.findall(r'\[([^\]]+)\]', message)

            # If no matches are found, the message does not concern a specific variable.
            if matches == []:
                continue

            variable = matches[0]
            message = message.split()

            if "constant" in message:
                removable_features.append(variable)
                print(f"Removed variable '{variable}' because it is constant.")
            if "zeros" in message:
                p_zeros = profile_json["variables"][variable]["p_zeros"]
                if p_zeros > self.params.clean.percentage_zeros_threshold:
                    removable_features.append(variable)
                    print(
                        f"Removed variable '{variable}' because % of zeros exceeds {self.params.clean.percentage_zeros_threshold*100}%."
                    )
            if "correlated" in message:
                try:
                    correlation_scores = correlations[variables.index(variable)]
                    for correlated_variable in correlation_scores:
                        if (
                            correlation_scores[correlated_variable]
                            > self.params.clean.input_max_correlation_threshold
                            and variable != correlated_variable
                            and variable != self.params.clean.target
                            and correlated_variable != self.params.clean.target
                            and variable not in removable_features
                        ):

                            removable_features.append(correlated_variable)
                            print(
                                f"Removed variable '{correlated_variable}' because of high correlation ({correlation_scores[correlated_variable]:.2f}) with variable '{variable}'."
                            )
                except:
                    # Pandas profiling might not be able to compute correlation
                    # score for some variables, for example some categorical
                    # variables.
                    print(f"{variable}: Could not find correlation score.")

        removable_features = list(set(removable_features))

        if self.params.clean.target in removable_features:
            print("Warning related to target variable. Check profile for details.")
            removable_features.remove(self.params.clean.target)

        return removable_features

def main():
    CleanStage().run()

if __name__ == "__main__":
    main()
