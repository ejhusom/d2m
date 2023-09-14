#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explain predictions of machine learning model.

Author:
    Erik Johannes Husom

Created:
    2022-11-28 mandag 16:01:00 

Notes:

    - https://github.com/slundberg/shap/issues/213

"""
import csv
import json
import os
import sys

import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import yaml

from codecarbon import track_emissions
from joblib import load
from matplotlib.colors import LinearSegmentedColormap
from lime import submodular_pick
from tensorflow.keras import models

from pipelinestage import PipelineStage
from config import config

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


class ExplainStage(PipelineStage):
    def __init__(self):
        super().__init__(stage_name="explain")

    def run(self):
        if not self.params.explain.generate_explanations:
            return 0

        if self.params.explain.seed is None:
            self.params.explain.seed = np.random.randint(0)

        # Load data
        self.train_data = np.load(config.DATA_COMBINED_TRAIN_PATH)
        self.X_train = self.train_data["X"]
        self.test_data = np.load(config.DATA_COMBINED_TEST_PATH)
        self.X_test = self.test_data["X"]
        self.y_test = self.test_data["y"]

        # Read name of input columns and convert to list
        self.input_columns = pd.read_csv(config.INPUT_FEATURES_PATH, index_col=0)
        self.input_columns = self.input_columns.values.flatten().tolist()

        if self.params.train.ensemble:
            self.explain_ensemble()
        else:
            feature_importances = []

            model = self.load_model(config.MODELS_FILE_PATH)

            if self.params.explain.explanation_method == "all":
                self.params.explain.explanation_method = config.EXPLANATION_METHODS

            if not isinstance(self.params.explain.explanation_method, list):
                self.params.explain.explanation_method = [
                    self.params.explain.explanation_method
                ]

            for explanation_method in self.params.explain.explanation_method:
                xai_values = self.explain_predictions(
                    model, explanation_method, make_plots=True
                )
                column_label = (
                    explanation_method + "_" + self.params.train.learning_method
                )
                feature_importance = get_feature_importance(
                    xai_values, label=column_label
                )

                # Scale feature importances to range of [0, 1]
                feature_importance = feature_importance.div(
                    feature_importance.sum(axis=0), axis=1
                )

                sorted_feature_importance = feature_importance.sort_values(
                    by=f"feature_importance_{column_label}", ascending=False
                )
                sorted_feature_importance.to_csv(
                    config.FEATURES_PATH / f"sorted_feature_importance_{column_label}.csv"
                )

                feature_importance = feature_importance.transpose()
                feature_importances.append(feature_importance)


            # Concat feature importance dataframe for all learning methods
            feature_importances = pd.concat(feature_importances)
            feature_importances.to_csv(config.FEATURES_PATH / "feature_importances.csv")

            pd.options.plotting.backend = "plotly"
            fig = feature_importances.plot.bar()
            fig.write_html(str(config.PLOTS_PATH / "feature_importances.html"))
            fig.show()

            # fig = feature_importances.transpose().plot.bar()
            # fig.write_html(str(config.PLOTS_PATH / "feature_importances.html"))
            # fig.show()

            # generate_explanation_report()

    def explain_ensemble(self):
        with open(config.ADEQUATE_MODELS_FILE_PATH, "r") as f:
            self.adequate_models = json.load(f)

        model_names = []
        adequate_methods = []

        for f in os.listdir(config.MODELS_PATH):
            if f.startswith("model"):
                model_names.append(f)

        model_names = sorted(model_names)

        feature_importances = []

        for name in model_names:
            if name in self.adequate_models.keys():
                method = os.path.splitext(name)[0].split("_")[-1]
                adequate_methods.append(method)

                model = self.load_model(config.MODELS_PATH / name,
                        method=method)

                print(f"Explaining {method}")

                # If user has chosen to use all explanation methods, set the
                # parameter accordingly
                if self.params.explain.explanation_method == "all":
                    self.params.explain.explanation_method = config.EXPLANATION_METHODS

                if not isinstance(self.params.explain.explanation_method, list):
                    self.params.explain.explanation_method = [
                        self.params.explain.explanation_method
                    ]
                for explanation_method in self.params.explain.explanation_method:
                    xai_values = self.explain_predictions(
                        model, explanation_method, make_plots=True
                    )
                    column_label = explanation_method + "_" + method
                    feature_importance = get_feature_importance(
                        xai_values, label=column_label
                    )

                    pos_feature_importance, neg_feature_importance = get_directional_feature_importance(xai_values, label=column_label)

                    # Scale feature importances to range of [0, 1]
                    feature_importance = feature_importance.div(
                        feature_importance.sum(axis=0), axis=1
                    )

                    sorted_feature_importance = feature_importance.sort_values(
                        by=f"feature_importance_{column_label}", ascending=False
                    )
                    sorted_pos_feature_importance = pos_feature_importance.sort_values(
                        ascending=False
                    )
                    sorted_neg_feature_importance = neg_feature_importance.sort_values(
                        ascending=False
                    )

                    sorted_feature_importance.to_csv(
                        config.FEATURES_PATH / f"sorted_feature_importance_{column_label}.csv"
                    )
                    sorted_pos_feature_importance.to_csv(
                        config.FEATURES_PATH / f"sorted_pos_feature_importance_{column_label}.csv"
                    )
                    sorted_neg_feature_importance.to_csv(
                        config.FEATURES_PATH / f"sorted_neg_feature_importance_{column_label}.csv"
                    )

                    feature_importance = feature_importance.transpose()
                    feature_importances.append(feature_importance)

        # Concat feature importance dataframe for all learning methods
        feature_importances = pd.concat(feature_importances)
        feature_importances.to_csv(config.FEATURES_PATH / "feature_importances.csv")

        pd.options.plotting.backend = "plotly"
        fig = feature_importances.plot.bar(
                labels=dict(index="", value="Feature importance (%)")
        )
        fig.write_html(str(config.PLOTS_PATH / "feature_importances.html"))
        fig.show()

        adequate_methods = sorted(adequate_methods)

        # generate_ensemble_explanation_tables(sorted_combined_feature_importances,
        #         adequate_methods)

        # Delete rows of the models in inadequate_models
        for index, row in feature_importances.iterrows():
            if index.split("_")[-1] not in adequate_methods:
                feature_importances.drop(index, inplace=True)

        feature_importances.to_csv(config.FEATURES_PATH /
                "feature_importances_adequate_models.csv")

    def explain_predictions(
        self,
        model,
        method="shap",
        make_plots=True,
    ):
        if method == "shap":
            xai_values = self.explain_predictions_shap(
                model,
                make_plots=make_plots,
            )
        elif method == "lime":
            xai_values = self.explain_predictions_lime(
                model,
                make_plots=make_plots,
            )
        else:
            raise NotImplementedError(
                f"Explanation method {method} is not implemented."
            )

        return xai_values

    def explain_predictions_shap(self, model, make_plots=False):
        X_test_summary = shap.sample(
            self.X_test,
            self.params.explain.number_of_summary_samples,
            random_state=self.params.explain.seed,
        )

        if self.params.train.learning_method in config.NON_SEQUENCE_LEARNING_METHODS or self.params.train.ensemble == True:
            if self.params.sequentialize.window_size > 1:
                input_columns_sequence = []

                for c in self.input_columns:
                    for i in range(self.params.sequentialize.window_size):
                        input_columns_sequence.append(c + f"_{i}")

                self.input_columns = input_columns_sequence

            # Extract a summary of the training inputs, to reduce the amount of
            # compute needed to use SHAP
            k = np.min([self.X_train.shape[0], 50])
            X_train_background = shap.kmeans(self.X_train, k)

            # Use a SHAP explainer on the summary of training inputs
            explainer = shap.KernelExplainer(model.predict, X_train_background)
            print(self.params.train.learning_method)
            # explainer = shap.TreeExplainer(model, X_train_background)

            # Single prediction explanation
            single_sample = self.X_test[0]
            single_shap_value = explainer.shap_values(single_sample)
            # Shap values for summary of test data
            shap_values = explainer.shap_values(X_test_summary)

            if isinstance(single_shap_value, list):
                single_shap_value = single_shap_value[0]
                shap_values = shap_values[0]

            if make_plots:
                # SHAP force plot: Single prediction
                shap_force_plot_single = shap.force_plot(
                    explainer.expected_value,
                    single_shap_value,
                    np.around(single_sample),
                    show=True,
                    feature_names=self.input_columns,
                )
                shap.save_html(
                    str(config.PLOTS_PATH) + "/shap_force_plot_single.html",
                    shap_force_plot_single,
                )

                # SHAP force plot: Multiple prediction
                shap_force_plot = shap.force_plot(
                    explainer.expected_value,
                    shap_values,
                    X_test_summary,
                    show=True,
                    feature_names=self.input_columns,
                )
                shap.save_html(
                    str(config.PLOTS_PATH) + "/shap_force_plot.html", shap_force_plot
                )

                # SHAP summary plot
                shap.summary_plot(
                    shap_values,
                    X_test_summary,
                    feature_names=self.input_columns,
                    plot_size=(8, 5),
                    show=False,
                    max_display=10,
                )
                plt.xticks(rotation=45)
                plt.tight_layout()

                plt.savefig(
                    config.PLOTS_PATH / "shap_summary_plot.png", bbox_inches="tight", dpi=300
                )

        else:
            if self.params.train.learning_method == "rnn":
                print("SHAP cannot be used with RNN models. Refer to the following issue: https://github.com/slundberg/shap/issues/2808")
                return 0

            # Extract a summary of the training inputs, to reduce the amount of
            # compute needed to use SHAP
            X_train_background = shap.sample(
                self.X_train,
                self.params.explain.number_of_background_samples,
                random_state=self.params.explain.seed,
            )

            # Use a SHAP explainer on the summary of training inputs
            explainer = shap.DeepExplainer(model, X_train_background)

            # Single prediction explanation
            single_sample = self.X_test[:1]
            single_shap_value = explainer.shap_values(single_sample)[0]
            shap_values = explainer.shap_values(X_test_summary)[0]

            if make_plots:
                # SHAP force plot: Single prediction
                shap_force_plot_single = shap.force_plot(
                    explainer.expected_value,
                    shap_values[0, :],
                    X_test_summary[0, :],
                    feature_names=self.input_columns,
                )
                shap.save_html(
                    str(config.PLOTS_PATH) + "/shap_force_plot_single.html",
                    shap_force_plot_single,
                )

                # Expand dimensions with 1 in order to plot. The built-in
                # image_plot of the shap library requires channel as one of the
                # dimensions in the input arrays. Therefore we add one dimension to
                # the arrays to make it seem like it is an image with one array.
                X_test_summary = np.expand_dims(X_test_summary, axis=3)
                shap_values = np.expand_dims(shap_values, axis=3)

                # SHAP image plot
                number_of_input_sequences = 5
                shap_image_plot = shap.image_plot(
                    shap_values[:number_of_input_sequences, :],
                    X_test_summary[:number_of_input_sequences, :],
                    show=False,
                )
                plt.savefig(
                    config.PLOTS_PATH / "shap_image_plot.png", bbox_inches="tight", dpi=300
                )

        shap_values = pd.DataFrame(shap_values, columns=self.input_columns).sort_index(
            axis=1
        )

        return shap_values

    def explain_predictions_lime(self, model, make_plots=False):
        if self.params.clean.classification:
            mode = "classification"
        else:
            mode = "regression"

        if (
                self.params.train.learning_method in config.SEQUENCE_LEARNING_METHODS 
                and self.params.sequentialize.window_size > 1 
                and self.params.train.ensemble == False
            ):

            lime_explainer = lime.lime_tabular.RecurrentTabularExplainer(
                self.X_test,
                feature_names=self.input_columns,
                mode=mode,
                discretize_continuous=False,
            )
        else:
            if self.params.sequentialize.window_size > 1:
                input_columns_sequence = []

                for c in self.input_columns:
                    for i in range(self.params.sequentialize.window_size):
                        input_columns_sequence.append(c + f"_{i}")

                self.input_columns = input_columns_sequence

            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_test,
                feature_names=self.input_columns,
                mode=mode,
                discretize_continuous=False,
            )

        sp_obj = lime.submodular_pick.SubmodularPick(
            lime_explainer,
            self.X_test,
            model.predict,
            sample_size=self.params.explain.number_of_background_samples,
            num_features=self.X_test.shape[-1],
        )

        # Making a dataframe of all the explanations of sampled points.
        xai_values = (
            pd.DataFrame([dict(this.as_list()) for this in sp_obj.explanations])
            .fillna(0)
            .sort_index(axis=1)
        )

        if make_plots:
            # Plotting the aggregate importances
            avg_xai_values = (
                np.abs(xai_values)
                .mean(axis=0)
                .sort_values(ascending=False)
                .head(25)
                .sort_values(ascending=True)
                .plot(kind="barh")
            )

            plt.savefig(
                config.PLOTS_PATH / "lime_summary_plot.png", bbox_inches="tight", dpi=300
            )

        return xai_values


def get_feature_importance(xai_values, label=""):
    # Modified from: https://github.com/slundberg/shap/issues/632

    vals = np.abs(xai_values).mean(0)
    feature_importance = pd.DataFrame(
        list(zip(xai_values.columns.tolist(), vals)),
        columns=["col_name", f"feature_importance_{label}"],
    )

    # feature_importance.sort_values(
    #     by=[f"feature_importance_{label}"], ascending=False, inplace=True
    # )

    feature_importance = feature_importance.set_index("col_name")

    return feature_importance

def get_directional_feature_importance(xai_values, label=""):

    print(xai_values)

    # Create separate dataframes for the positive and negative XAI values for
    # each feature.
    xai_values_pos = xai_values.copy()
    xai_values_neg = xai_values.copy()

    # Remove negative values in the positive data frame and vice versa.
    xai_values_pos[xai_values_pos < 0] = 0
    xai_values_neg[xai_values_neg > 0] = 0

    # Calculate the mean SHAP values for pos/neg separately.
    pos_feature_importance = xai_values_pos.mean()
    neg_feature_importance = xai_values_neg.mean()

    # Calculate the sum of absolute average positive and negative SHAP values
    total_impact = pos_feature_importance.abs() + neg_feature_importance.abs()

    # Get the 10 features with the highest impact
    top_features = total_impact.nlargest(10).index

    # Extract the positive and negative SHAP values for the top features
    top_avg_shap_positive = pos_feature_importance[top_features]
    top_avg_shap_negative = neg_feature_importance[top_features]

    # Create a new figure and set its size
    plt.figure(figsize=(10, 8))

    # Create an array for the positions of the bars on the y-axis
    ind = np.arange(len(top_features))

    # Width of a bar
    width = 0.35

    # Plotting
    plt.barh(ind, top_avg_shap_positive.values, width, color='b', label='positive')
    plt.barh(ind, top_avg_shap_negative.values, width, color='r', label='negative')

    # Adding feature names as y labels
    plt.yticks(ind, top_features)

    plt.axvline(0, color='k')
    plt.xlabel('Average feature importance')
    plt.title('Average positive and negative feature importance for the top 10 impactful features')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig(config.PLOTS_PATH / f"directional_feature_importance_{label}.png")

    pos_feature_importance.name = "feature_importance_" + label
    neg_feature_importance.name = "feature_importance_" + label

    return pos_feature_importance, neg_feature_importance

def main():
    ExplainStage().run()

if __name__ == "__main__":
    main()
