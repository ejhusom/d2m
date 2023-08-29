#!/usr/bin/env python3
"""Global parameters for project.

Example:

    >>> from config import config
    >>> some_path = config.PARAMS_FILE_PATH
    >>> file = config.DATA_PATH / "filename.txt"

Author:   Erik Johannes Husom
Created:  2020-09-16

"""
from pathlib import Path

class Config:
    def __init__(self):
        # PARAMETERS
        self.DL_METHODS = ["dnn", "cnn", "rnn"]
        self.NON_DL_METHODS = [
            "dt", "rf", "explainableboosting", "xgboost", "xgb", "lgbm",
            "lda", "qda", "svm", "linearregression", "ridgeregression",
            "lasso", "lars", "bayesianridge", "ardregression", "elasticnet",
            "sgd", "kneighbors", "kn", "gb", "gradientboosting"
        ]
        self.SEQUENCE_LEARNING_METHODS = ["cnn", "rnn"]
        self.NON_SEQUENCE_LEARNING_METHODS = [method for method in self.NON_DL_METHODS if method not in self.SEQUENCE_LEARNING_METHODS]
        self.NON_SEQUENCE_LEARNING_METHODS += [method for method in self.DL_METHODS if method not in self.SEQUENCE_LEARNING_METHODS]
        self.METHODS_IN_ENSEMBLE = [
            "dt", "rf", "gb", "xgb", "sgd", "lgbm"
        ]
        self.EXPLANATION_METHODS = ["shap", "lime"]

        # PATHS
        self.PARAMS_FILE_PATH = Path("./params.yaml")
        self.ASSETS_PATH = Path("./assets")
        self.PROFILE_PATH = self.ASSETS_PATH / "profile"
        self.PROFILE_HTML_PATH = self.PROFILE_PATH / "profile.html"
        self.PROFILE_JSON_PATH = self.PROFILE_PATH / "profile.json"
        self.FEATURES_PATH = self.ASSETS_PATH / "features"
        self.INPUT_FEATURES_PATH = self.FEATURES_PATH / "input_columns.csv"
        self.INPUT_FEATURES_SEQUENCE_PATH = self.FEATURES_PATH / "input_columns_sequence.csv"
        self.OUTPUT_FEATURES_PATH = self.FEATURES_PATH / "output_columns.csv"
        self.REMOVABLE_FEATURES = self.FEATURES_PATH / "removable_features.csv"
        self.DATA_PATH = self.ASSETS_PATH / "data"
        self.DATA_PATH_RAW = self.DATA_PATH / "raw"
        self.DATA_FEATURIZED_PATH = self.DATA_PATH / "featurized"
        self.DATA_CLEANED_PATH = self.DATA_PATH / "cleaned"
        self.DATA_SEQUENTIALIZED_PATH = self.DATA_PATH / "sequentialized"
        self.DATA_SPLIT_PATH = self.DATA_PATH / "split"
        self.DATA_SCALED_PATH = self.DATA_PATH / "scaled"
        self.DATA_COMBINED_PATH = self.DATA_PATH / "combined"
        self.DATA_COMBINED_TRAIN_PATH = self.DATA_COMBINED_PATH / "train.npz"
        self.DATA_COMBINED_TEST_PATH = self.DATA_COMBINED_PATH / "test.npz"
        self.MODELS_PATH = self.ASSETS_PATH / "models"
        self.MODELS_FILE_PATH = self.MODELS_PATH / "model.h5"
        self.API_MODELS_PATH = self.ASSETS_PATH / "models_api.json"
        self.METRICS_PATH = self.ASSETS_PATH / "metrics"
        self.METRICS_FILE_PATH = self.METRICS_PATH / "metrics.json"
        self.PREDICTIONS_PATH = self.ASSETS_PATH / "predictions"
        self.PREDICTIONS_FILE_PATH = self.PREDICTIONS_PATH / "predictions.csv"
        self.PLOTS_PATH = self.ASSETS_PATH / "plots"
        self.PREDICTION_PLOT_PATH = self.PLOTS_PATH / "prediction.png"
        self.INTERVALS_PLOT_PATH = self.PLOTS_PATH / "intervals.png"
        self.TRAININGLOSS_PLOT_PATH = self.PLOTS_PATH / "trainingloss.png"
        self.SCALER_PATH = self.ASSETS_PATH / "scalers"
        self.INPUT_SCALER_PATH = self.SCALER_PATH / "input_scaler.z"
        self.OUTPUT_SCALER_PATH = self.SCALER_PATH / "output_scaler.z"
        self.OUTPUT_PATH = self.ASSETS_PATH / "output"
        self.ADEQUATE_MODELS_PATH = self.ASSETS_PATH / "adequate_models"
        self.ADEQUATE_MODELS_FILE_PATH = self.ADEQUATE_MODELS_PATH / "adequate_models.json"

        self._init_paths()

    def _init_paths(self):
        """Create directories if they don't exist."""
        directories = [
            self.ASSETS_PATH,
            self.PROFILE_PATH,
            self.FEATURES_PATH,
            self.DATA_PATH,
            self.DATA_PATH_RAW,
            self.DATA_FEATURIZED_PATH,
            self.DATA_CLEANED_PATH,
            self.DATA_SEQUENTIALIZED_PATH,
            self.DATA_SPLIT_PATH,
            self.DATA_SCALED_PATH,
            self.DATA_COMBINED_PATH,
            self.MODELS_PATH,
            self.METRICS_PATH,
            self.PREDICTIONS_PATH,
            self.PLOTS_PATH,
            self.SCALER_PATH,
            self.OUTPUT_PATH,
            self.ADEQUATE_MODELS_PATH
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Instantiate a single configuration object to use throughout your application
config = Config()

