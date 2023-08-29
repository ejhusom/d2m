#!/usr/bin/env python3
"""Evaluate deep learning model.

Author:
    Erik Johannes Husom

Created:
    2020-09-17

"""
import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import seaborn as sn
import tensorflow as tf
import yaml
from codecarbon import track_emissions
from joblib import load
from plotly.subplots import make_subplots
from sklearn.base import RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    explained_variance_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras import metrics, models
from scipy.sparse import coo_matrix
from sklearn.utils.multiclass import unique_labels

import neural_networks as nn

import neural_networks as nn
from config import config
from pipelinestage import PipelineStage

class EvaluateStage(PipelineStage):

    def __init__(self):
        super().__init__(stage_name="evaluate")

    def run(self):

        if self.params.evaluate.performance_metric == "auto":
            if self.params.clean.classification:
                self.params.evaluate.performance_metric = "accuracy"
            else:
                self.params.evaluate.performance_metric = "r2"

        if self.params.evaluate.threshold_for_ensemble_models == "auto":
            if self.params.clean.classification:
                self.params.evaluate.threshold_for_ensemble_models = 0.75
            else:
                self.params.evaluate.threshold_for_ensemble_models = 0.5

        test = np.load(config.DATA_COMBINED_TEST_PATH)
        X_test = test["X"]
        y_test = test["y"]
        y_pred_uncertainty = None

        if self.params.evaluate.show_inputs:
            inputs = X_test
        else:
            inputs = None

        pd.DataFrame(y_test).to_csv(config.PREDICTIONS_PATH / "true_values.csv")

        if self.params.train.ensemble:
            model_names = []
            y_preds = {}
            metrics = []
            # info = ". "

            for f in os.listdir(config.MODELS_PATH):
                if f.startswith("model"):
                    model_names.append(f)

            model_names = sorted(model_names)

            for name in model_names:
                method = os.path.splitext(name)[0].split("_")[-1]
                model = self.load_model(config.MODELS_PATH / name, method)

                y_pred = model.predict(X_test)
                y_preds[method] = y_pred

            adequate_models = {}

            if self.params.clean.classification:

                if self.params.clean.onehot_encode_target:
                    y_test = np.argmax(y_test, axis=-1)

                metrics = {}

                for name in model_names:
                    method = os.path.splitext(name)[0].split("_")[-1]
                    y_pred = y_preds[method]
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    print(f"{name} precision: {precision}")
                    print(f"{name} recall: {recall}")
                    print(f"{name} F1: {f1}")
                    print(f"{name} accuracy: {accuracy}")
                    metrics[name] = accuracy

                    if accuracy >= self.params.evaluate.threshold_for_ensemble_models:
                        adequate_models[name] = accuracy

                    y_preds[method + f" ({accuracy:.2f})"] = y_preds.pop(method)

                # plot_prediction(y_test, y_pred, info="Accuracy: {})".format(accuracy))
                # plot_confusion(y_test, y_pred)

                with open(config.METRICS_FILE_PATH, "w") as f:
                    json.dump(metrics, f)

            # Regression:
            else:
                metrics = {}

                for name in model_names:
                    method = os.path.splitext(name)[0].split("_")[-1]
                    print(f"{name}, {method}")
                    y_pred = y_preds[method]
                    metrics[name] = {}

                    mse = mean_squared_error(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    metrics[name]["mse"] = mse
                    metrics[name]["rmse"] = rmse
                    metrics[name]["mape"] = mape
                    metrics[name]["r2"] = r2
                    # metrics[name] = r2

                    # plot_prediction(y_test, y_pred, inputs=inputs, info=f"(R2: {r2:.2f})")
                    # plot_true_vs_pred(y_test, y_pred)

                    # print("MSE: {}".format(mse))
                    # print("RMSE: {}".format(rmse))
                    # print("MAPE: {}".format(mape))
                    # print(f"{name} R2: {r2:.3f}")

                    # info += f"{name} {r2:.2f}. "
                    y_preds[method + f" ({r2:.2f})"] = y_preds.pop(method)

                    for n in y_preds:
                        print(n)
                    print("======")

                    if metrics[name][self.params.evaluate.performance_metric] >= self.params.evaluate.threshold_for_ensemble_models:
                        adequate_models[name] = metrics[name][self.params.evaluate.performance_metric]


                # # Only plot predicted sequences if the output samples are sequences.
                # if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                #     plot_sequence_predictions(y_test, y_pred)

                # with open(config.METRICS_FILE_PATH, "w") as f:
                #     json.dump(dict(mse=mse, rmse=rmse, mape=mape, r2=r2), f)
                with open(config.METRICS_FILE_PATH, "w") as f:
                    json.dump(metrics, f)


                with open(config.ADEQUATE_MODELS_PATH / "adequate_models.json", "w") as f:
                    json.dump(adequate_models, f)

                # save_predictions(pd.DataFrame(y_pred))

            plot_prediction(y_test, y_preds, inputs=inputs, info="ensemble")

            return 0

        # pandas data frame to store predictions and ground truth.
        df_predictions = None

        y_pred = None

        if self.params.train.learning_method in config.NON_DL_METHODS:
            model = load(config.MODELS_FILE_PATH)
            y_pred = model.predict(X_test)
        else:
            model = models.load_model(config.MODELS_FILE_PATH)

            if self.params.evaluate.dropout_uncertainty_estimation and not self.params.train.ensemble:
                predictions = []

                for i in range(self.params.evaluate.uncertainty_estimation_sampling_size):
                    predictions.append(model(X_test, training=True))

                predictions = np.stack(predictions, -1)
                mean = np.mean(predictions, axis=-1)

                if self.params.clean.classification:
                    entropy = - np.sum(predictions * np.log(predictions + 1e-15), axis=-1)
                    uncertainty = entropy
                else:
                    uncertainty = np.std(predictions, axis=-1)

                y_pred = mean
                y_pred_uncertainty = uncertainty
                pd.DataFrame(y_pred_uncertainty).to_csv(config.PREDICTIONS_PATH /
                        "predictions_uncertainty.csv")
            else:
                y_pred = model.predict(X_test)

        # Check if the shape of y_pred matches y_test, and if not, reshape y_pred
        # to match y_test.
        if y_pred.shape != y_test.shape:
            if len(y_test.shape) == 1:
                y_pred = y_pred.reshape(-1)
            elif len(y_test.shape) == 2:
                y_pred = y_pred.reshape(-1, y_test.shape[1])
            else:
                raise ValueError("y_test has more than 2 dimensions.")

        if self.params.clean.onehot_encode_target:
            y_pred = np.argmax(y_pred, axis=-1)
        elif self.params.clean.classification:
            y_pred = np.array((y_pred > 0.5), dtype=np.int)

        if self.params.clean.classification:

            if self.params.clean.onehot_encode_target:
                y_test = np.argmax(y_test, axis=-1)

            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy}")

            plot_prediction(y_test, y_pred, info="Accuracy: {})".format(accuracy))

            plot_confusion(y_test, y_pred, y_pred_uncertainty)

            with open(config.METRICS_FILE_PATH, "w") as f:
                json.dump(dict(accuracy=accuracy), f)

        # Regression:
        else:
            print(y_test)
            print(y_pred)
            print(y_test.shape)
            print(y_pred.shape)
            print("========")
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            plot_prediction(y_test, y_pred, inputs=inputs, info=f"(R2: {r2:.2f})",
                            y_pred_uncertainty=y_pred_uncertainty)
            plot_true_vs_pred(y_test, y_pred)

            print("MSE: {}".format(mse))
            print("RMSE: {}".format(rmse))
            print("MAPE: {}".format(mape))
            print("R2: {}".format(r2))

            # Only plot predicted sequences if the output samples are sequences.
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                plot_sequence_predictions(y_test, y_pred)

            with open(config.METRICS_FILE_PATH, "w") as f:
                json.dump(dict(mse=mse, rmse=rmse, mape=mape, r2=r2), f)

        # Print feature importances of the ML algorithm supports it.
        try:
            feature_importances = model.feature_importances_
            imp = list()
            for i, f in enumerate(feature_importances):
                imp.append((f, i))

            sorted_feature_importances = sorted(imp)[::-1]
            input_columns = pd.read_csv(config.INPUT_FEATURES_PATH, header=None)

            print("-------------------------")
            print("Feature importances:")

            for i in range(len(sorted_feature_importances)):
                print(
                    f"Feature: {input_columns.iloc[i,0]}. Importance: {feature_importances[i]:.2f}"
                )

            print("-------------------------")
        except:
            pass

        save_predictions(pd.DataFrame(y_pred))

def plot_confusion(y_test, y_pred, y_pred_uncertainty=None):
    """Plotting confusion matrix of a classification model."""

    output_columns = np.array(pd.read_csv(config.OUTPUT_FEATURES_PATH, index_col=0)).reshape(
        -1
    )

    n_output_cols = len(output_columns)
    indeces = np.arange(0, n_output_cols, 1)

    confusion = confusion_matrix(y_test, y_pred, normalize="true")
    # labels=indeces)

    print(confusion)

    df_confusion = pd.DataFrame(confusion)

    df_confusion.index.name = "True"
    df_confusion.columns.name = "Pred"
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_confusion, cmap="Blues", annot=True, annot_kws={"size": 16})
    plt.savefig(config.PLOTS_PATH / "confusion_matrix.png")

    if y_pred_uncertainty is not None:

        # Take the average of the entropy per class
        y_pred_uncertainty = np.mean(y_pred_uncertainty, axis=1)

        cm = coo_matrix(
                (y_pred_uncertainty, (y_test, y_pred)),
                shape=(n_labels, n_labels)
        )

        combined_arr = np.stack((y_test, y_pred), axis=1)

        unique_predictions, counts = np.unique(combined_arr, axis=0, return_counts=True)
        u_true = unique_predictions[:,0]
        u_pred = unique_predictions[:,1]

        cm_count = coo_matrix(
                (counts, (u_true, u_pred)),
                shape=(n_labels, n_labels)
        )

        cm = cm / cm_count
        
        cm = pd.DataFrame(cm)

        df_confusion.index.name = "True"
        df_confusion.columns.name = "Pred"

        plt.figure(figsize=(10, 7))
        sn.heatmap(
                cm, 
                cmap="Reds", 
                annot=df_confusion, 
                # annot=True,
                annot_kws={"size": 14},
                xticklabels=labels,
                yticklabels=labels,
        )
        plt.tight_layout()
        plt.savefig(config.PLOTS_PATH / "probablistic_confusion_matrix.png")
        # plt.show()

def save_predictions(df_predictions):
    """Save the predictions along with the ground truth as a csv file.

    Args:
        df_predictions_true (pandas dataframe): pandas data frame with the predictions and ground truth values.

    """


    df_predictions.to_csv(config.PREDICTIONS_FILE_PATH, index=False)


def plot_confidence_intervals(df):
    """Plot the confidence intervals generated with conformal prediction.

    Args:
        df (pandas dataframe): pandas data frame.

    """


    x = [x for x in range(1, df.shape[0] + 1, 1)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=df["predicted"], name="predictions"))

    fig.add_trace(
        go.Scatter(
            name="Upper Bound",
            x=x,
            y=df["upper_bound"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Lower Bound",
            x=x,
            y=df["lower_bound"],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
            showlegend=False,
        )
    )

    fig.write_html(str(config.PLOTS_PATH / "intervals.html"))


def plot_true_vs_pred(y_true, y_pred):

    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.savefig(config.PLOTS_PATH / "true_vs_pred.png")


def plot_prediction(y_true, y_pred, inputs=None, info="", y_pred_uncertainty=None):
    """Plot the prediction compared to the true targets.

    Args:
        y_true (array): True targets.
        y_pred (array): Predicted targets.
        include_input (bool): Whether to include inputs in plot. Default=True.
        inputs (array): Inputs corresponding to the targets passed. If
            provided, the inputs will be plotted together with the targets.
        info (str): Information to include in the title string.
        y_pred_uncertainty (array): Uncertainty of the true targets.

    Returns:
        fig (plotly figure): Plotly figure.

    """

    if isinstance(y_pred, dict):
        ensemble = True
    else:
        ensemble = False


    x = np.linspace(0, y_true.shape[0] - 1, y_true.shape[0])
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if len(y_true.shape) > 1:
        y_true = y_true[:, -1].reshape(-1)
    if ensemble:
        for arr in y_pred:
            if len(y_pred[arr].shape) > 1:
                y_pred[arr] = y_pred[arr][:, -1].reshape(-1)
    else:
        if len(y_pred.shape) > 1:
            y_pred = y_pred[:, -1].reshape(-1)
        if y_pred_uncertainty is not None and len(y_pred_uncertainty.shape) > 1:
            y_pred_uncertainty = y_pred_uncertainty[:, -1].reshape(-1)

    fig.add_trace(
        go.Scatter(x=x, y=y_true, name="true"),
        secondary_y=False,
    )

    if ensemble:
        for arr in y_pred:
            fig.add_trace(
                go.Scatter(x=x, y=y_pred[arr], name=arr),
                secondary_y=False,
            )
    else:
        fig.add_trace(
            go.Scatter(x=x, y=y_pred, name="pred"),
            secondary_y=False,
        )

    if inputs is not None:
        input_columns = pd.read_csv(config.INPUT_FEATURES_PATH, index_col=0)
        input_columns = [feature for feature in input_columns["0"]]

        if len(inputs.shape) == 3:
            n_features = inputs.shape[-1]
        elif len(inputs.shape) == 2:
            n_features = len(input_columns)

        for i in range(n_features):

            if len(inputs.shape) == 3:
                fig.add_trace(
                    go.Scatter(x=x, y=inputs[:, -1, i], name=input_columns[i]),
                    secondary_y=True,
                )
            elif len(inputs.shape) == 2:
                fig.add_trace(
                    go.Scatter(x=x, y=inputs[:, i - n_features], name=input_columns[i]),
                    secondary_y=True,
                )

    if y_pred_uncertainty is not None:
        fig.add_trace(
            go.Scatter(
                name="Uncertainty bottom",
                x=x,
                y=y_pred - 1.96*y_pred_uncertainty,
                line=dict(width=0),
                showlegend=False,
            ),
        )
        fig.add_trace(
            go.Scatter(
                name="Uncertainty",
                x=x,
                y=y_pred + 1.96*y_pred_uncertainty,
                line=dict(width=0),
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=True,
            ),
        )

    fig.update_layout(title_text="True vs pred " + info)
    fig.update_xaxes(title_text="time step")
    fig.update_yaxes(title_text="target unit", secondary_y=False)
    fig.update_yaxes(title_text="scaled units", secondary_y=True)

    fig.write_html(str(config.PLOTS_PATH / "prediction.html"))

    # fig.update_traces(line=dict(width=0.8))
    # fig.write_image("plot.pdf", height=270, width=560)
    # fig.write_image("plot.png", height=270, width=560, scale=10)

    return fig


def plot_sequence_predictions(y_true, y_pred):
    """
    Plot the prediction compared to the true targets.

    """

    target_size = y_true.shape[-1]
    pred_curve_step = target_size

    pred_curve_idcs = np.arange(0, y_true.shape[0], pred_curve_step)
    # y_indeces = np.arange(0, y_true.shape[0]-1, 1)
    y_indeces = np.linspace(0, y_true.shape[0] - 1, y_true.shape[0])

    n_pred_curves = len(pred_curve_idcs)

    fig = go.Figure()

    y_true_df = pd.DataFrame(y_true[:, 0])

    fig.add_trace(go.Scatter(x=y_indeces, y=y_true[:, 0].reshape(-1), name="true"))

    predictions = []

    for i in pred_curve_idcs:
        indeces = y_indeces[i : i + target_size]

        if len(indeces) < target_size:
            break

        y_pred_df = pd.DataFrame(y_pred[i, :], index=indeces)

        predictions.append(y_pred_df)

        fig.add_trace(
            go.Scatter(
                x=indeces, y=y_pred[i, :].reshape(-1), showlegend=False, mode="lines"
            )
        )


    fig.write_html(str(config.PLOTS_PATH / "prediction_sequences.html"))

def main():
    EvaluateStage().run()

if __name__ == "__main__":
    main()
