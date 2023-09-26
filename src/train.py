#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train deep learning model to estimate power from breathing data.


Author:
    Erik Johannes Husom

Created:
    2020-09-16  

"""
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from codecarbon import track_emissions
from joblib import dump
from lightgbm import LGBMClassifier, LGBMRegressor
from keras_tuner import HyperParameters
from keras_tuner.tuners import BayesianOptimization, Hyperband, RandomSearch
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    SGDClassifier,
    SGDRegressor,
    Ridge,
    RidgeCV,
    Lasso,
    Lars,
    BayesianRidge,
    ARDRegression,
    ElasticNet,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

import neural_networks as nn
from config import config
from pipelinestage import PipelineStage


@track_emissions(project_name="train")
class TrainStage(PipelineStage):

    def __init__(self):
        super().__init__(stage_name="train")

    def run(self):
        output_columns = np.array(pd.read_csv(config.OUTPUT_FEATURES_PATH, index_col=0)).reshape(
            -1
        )

        n_output_cols = len(output_columns)

        # Load training set
        train_data = np.load(config.DATA_COMBINED_TRAIN_PATH)

        X_train = train_data["X"]
        y_train = train_data["y"]

        n_features = X_train.shape[-1]
        hist_size = X_train.shape[-2]
        self.params.sequentialize.target_size = y_train.shape[-1]

        if self.params.clean.classification:
            if self.params.clean.onehot_encode_target:
                output_activation = "softmax"
                loss = "categorical_crossentropy"
            else:
                output_activation = "sigmoid"
                loss = "binary_crossentropy"
            output_length = n_output_cols
            metrics = "accuracy"
            monitor_metric = "accuracy"
        else:
            output_activation = "linear"
            output_length = self.params.sequentialize.target_size
            loss = "mse"
            metrics = "mse"
            monitor_metric = "loss"

        # Create an ensemble
        if self.params.train.ensemble:
            # model0 = nn.dnn(
            #     n_features,
            #     output_length=output_length,
            #     activation_function=self.params.train.activation_function,
            #     output_activation=output_activation,
            #     n_layers=self.params.train.n_layers,
            #     n_nodes=self.params.train.n_neurons,
            #     loss=loss,
            #     metrics=metrics,
            #     dropout=self.params.train.dropout,
            #     seed=self.params.train.seed
            # )
            if self.params.clean.classification:
                model0 = DecisionTreeClassifier()
                model1 = RandomForestClassifier()
                model2 = GradientBoostingClassifier()
                model3 = xgb.XGBClassifier()
                model4 = SGDClassifier()
                model5 = LGBMClassifier()
                # model0 = SVC()
                # model3 = KNeighborsClassifier()
            else:
                model0 = DecisionTreeRegressor()
                model1 = RandomForestRegressor()
                model2 = GradientBoostingRegressor()
                model3 = xgb.XGBRegressor()
                model4 = SGDRegressor()
                model5 = LGBMRegressor()
                # model0 = SVR()
                # model3 = KNeighborsRegressor()

            models = [
                    model0,
                    model1,
                    model2,
                    model3,
                    model4,
                    model5,
                    # model6,
            ]


            for name, model in zip(config.METHODS_IN_ENSEMBLE, models):
                if name in config.DL_METHODS:
                    history = model0.fit(
                        X_train,
                        y_train,
                        epochs=self.params.train.n_epochs,
                        batch_size=self.params.train.batch_size,
                        validation_split=0.25,
                    )
                    model.save(config.MODELS_PATH  / f"model_{name}.h5")
                else:
                    model.fit(X_train, y_train.ravel())
                    dump(model, config.MODELS_PATH / f"model_{name}.h5")

            return 0


        # Build model
        if self.params.train.learning_method in config.DL_METHODS and self.params.train.hyperparameter_tuning:

            # In order to perform model tuning, any old model_tuning results must
            # be deleted.
            if os.path.exists("model_tuning"):
                shutil.rmtree("model_tuning")

            if self.params.train.learning_method == "lstm":
                hypermodel = nn.LSTMHyperModel(
                    hist_size, n_features, loss=loss, metrics=metrics
                )
            elif self.params.train.learning_method == "cnn":
                hypermodel = nn.CNNHyperModel(
                    hist_size, n_features, loss=loss, metrics=metrics
                )
            else:
                hypermodel = nn.SequentialHyperModel(n_features, loss=loss, metrics=metrics)

            hypermodel.build(HyperParameters())
            tuner = BayesianOptimization(
                hypermodel,
                objective="val_loss",
                directory="model_tuning",
            )
            tuner.search_space_summary()
            tuner.search(
                X_train,
                y_train,
                epochs=200,
                batch_size=self.params.train.batch_size,
                validation_split=0.2,
            )
            tuner.results_summary()

            model = tuner.get_best_models()[0]

        elif self.params.train.learning_method.startswith("dnn"):
            build_model = getattr(nn, self.params.train.learning_method)
            model = build_model(
                n_features,
                output_length=output_length,
                activation_function=self.params.train.activation_function,
                output_activation=output_activation,
                n_layers=self.params.train.n_layers,
                n_nodes=self.params.train.n_neurons,
                loss=loss,
                metrics=metrics,
                dropout=self.params.train.dropout,
                seed=self.params.train.seed
            )
        elif self.params.train.learning_method.startswith("cnn"):
            hist_size = X_train.shape[-2]
            build_model = getattr(nn, self.params.train.learning_method)
            model = build_model(
                hist_size,
                n_features,
                output_length=output_length,
                kernel_size=self.params.train.kernel_size,
                activation_function=self.params.train.activation_function,
                output_activation=output_activation,
                loss=loss,
                metrics=metrics,
                n_layers=self.params.train.n_layers,
                n_filters=self.params.train.n_neurons,
                maxpooling=self.params.train.maxpooling,
                maxpooling_size=self.params.train.maxpooling_size,
                dropout=self.params.train.dropout,
                n_dense_layers=self.params.train.n_flattened_layers,
                n_nodes=self.params.train.n_flattened_nodes,
                seed=self.params.train.seed
            )
        elif self.params.train.learning_method.startswith("rnn"):
            hist_size = X_train.shape[-2]
            build_model = getattr(nn, self.params.train.learning_method)
            model = build_model(
                hist_size,
                n_features,
                output_length=output_length,
                unit_type=self.params.train.unit_type.lower(),
                activation_function=self.params.train.activation_function,
                output_activation=output_activation,
                loss=loss,
                metrics=metrics,
                n_layers=self.params.train.n_layers,
                n_units=self.params.train.n_neurons,
                dropout=self.params.train.dropout,
                n_dense_layers=self.params.train.n_flattened_layers,
                n_nodes=self.params.train.n_flattened_nodes,
                seed=self.params.train.seed
            )
        elif self.params.train.learning_method.startswith("transformer"):
            hist_size = X_train.shape[-2]
            build_model = getattr(nn, self.params.train.learning_method)
            model = build_model(
                input_size_x=hist_size,
                input_size_y=n_features,
                output_length=output_length,
                head_size=self.params.train.head_size,
                n_heads=self.params.train.n_heads,
                ff_dim=self.params.train.ff_dim,
                n_transformer_blocks=self.params.train.n_transformer_blocks,
                mlp_layers=self.params.train.n_flattened_layers,
                mlp_units=self.params.train.n_flattened_nodes,
                dropout=self.params.train.dropout,
                mlp_dropout=self.params.train.dropout,
                activation_function=self.params.train.activation_function,
                output_activation=output_activation,
                loss=loss,
                metrics=metrics,
                # hist_size,
                # n_features,
                # output_length=output_length,
                # activation_function=self.params.train.activation_function,
                # output_activation=output_activation,
                # loss=loss,
                # metrics=metrics,
                # n_layers=self.params.train.n_layers,
                # n_heads=self.params.train.n_heads,
                # n_units=self.params.train.n_neurons,
                # n_dense_layers=self.params.train.n_flattened_layers,
                # n_nodes=self.params.train.n_flattened_nodes,
                # dropout=self.params.train.dropout,
                # seed=self.params.train.seed
            )
        elif self.params.train.learning_method == "dt":
            if self.params.clean.classification:
                model = DecisionTreeClassifier()
            else:
                model = DecisionTreeRegressor()
        elif self.params.train.learning_method == "rf":
            if self.params.clean.classification:
                model = RandomForestClassifier()
            else:
                model = RandomForestRegressor()
            if self.params.train.hyperparameter_tuning:
                model = RandomizedSearchCV(
                    model,
                    {
                        "max_depth": [2, 4, 6, 8],
                        "n_estimators": [50, 100, 200, 400, 800, 1000],
                        "min_samples_split": [2, 4, 6, 8, 10],
                        "min_samples_leaf": [1, 3, 5],
                    },
                    verbose=2,
                )
        elif self.params.train.learning_method == "kneighbors" or self.params.train.learning_method == "kn":
            if self.params.clean.classification:
                model = KNeighborsClassifier()
            else:
                model = KNeighborsRegressor()
            if self.params.train.hyperparameter_tuning:
                model = RandomizedSearchCV(
                    model,
                    {
                        "n_neighbors": [2, 4, 5, 6, 10, 15, 20, 30],
                        "weights": ["uniform", "distance"],
                        "leaf_size": [10, 30, 50, 80, 100],
                        "algorithm": ["ball_tree", "kd_tree", "brute"],
                    },
                    verbose=2,
                )
        elif self.params.train.learning_method == "gradientboosting" or self.params.train.learning_method == "gb":
            if self.params.clean.classification:
                model = GradientBoostingClassifier()
            else:
                model = GradientBoostingRegressor()
        elif self.params.train.learning_method == "xgboost":
            if self.params.clean.classification:
                model = xgb.XGBClassifier()
            else:
                if self.params.sequentialize.target_size > 1:
                    model = MultiOutputRegressor(xgb.XGBRegressor())
                else:
                    model = xgb.XGBRegressor()
            if self.params.train.hyperparameter_tuning:
                model = RandomizedSearchCV(
                    model,
                    {
                        "max_depth": [2, 4, 6, 8],
                        "n_estimators": [50, 100, 200, 400, 800, 1000],
                        "learning_rate": [0.3, 0.1, 0.001, 0.0001],
                    },
                    verbose=2,
                )
        elif self.params.train.learning_method.lower() == "explainableboosting":
            if self.params.clean.classification:
                model = ExplainableBoostingClassifier(max_rounds=2)
            else:
                model = ExplainableBoostingRegressor()
        elif self.params.train.learning_method == "linearregression":
            if self.params.clean.classification:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with regression."
                )
            else:
                model = LinearRegression()
        elif self.params.train.learning_method == "ridgeregression":
            if self.params.clean.classification:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with regression."
                )
            else:
                model = Ridge()
        elif self.params.train.learning_method == "lasso":
            if self.params.clean.classification:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with regression."
                )
            else:
                model = Lasso()
        elif self.params.train.learning_method == "lars":
            if self.params.clean.classification:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with regression."
                )
            else:
                model = Lars()
        elif self.params.train.learning_method == "bayesianridge":
            if self.params.clean.classification:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with regression."
                )
            else:
                model = BayesianRidge()
        elif self.params.train.learning_method == "ardregression":
            if self.params.clean.classification:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with regression."
                )
            else:
                model = ARDRegression()
        elif self.params.train.learning_method == "elasticnet":
            if self.params.clean.classification:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with regression."
                )
            else:
                model = ElasticNet()
        elif self.params.train.learning_method == "lda":
            if self.params.clean.classification:
                model = LinearDiscriminantAnalysis()
            else:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with classification."
                )
        elif self.params.train.learning_method == "sgd":
            if self.params.clean.classification:
                model = SGDClassifier()
            else:
                model = SGDRegressor()
        elif self.params.train.learning_method == "qda":
            if self.params.clean.classification:
                model = QuadraticDiscriminantAnalysis()
            else:
                raise ValueError(
                    f"Learning method {self.params.train.learning_method} only works with classification."
                )
        elif self.params.train.learning_method == "svm":
            if self.params.clean.classification:
                model = SVC()
            else:
                model = SVR()
            if self.params.train.hyperparameter_tuning:
                model = RandomizedSearchCV(
                    model,
                    {
                        "kernel": ["linear", "poly", "rbf"],
                        "degree": [1, 3, 5],
                        "max_iter": [1, 5, 10],
                    },
                )
        elif self.params.train.learning_method == "brnn":
            model = nn.brnn(
                data_size=X_train.shape[0],
                window_size=X_train.shape[1],
                feature_size=X_train.shape[2],
                batch_size=self.params.train.batch_size,
                hidden_size=10,
            )  # TODO: Make this into a parameter
        elif self.params.train.learning_method == "bcnn":
            model = nn.bcnn(
                data_size=X_train.shape[0],
                window_size=X_train.shape[1],
                feature_size=X_train.shape[2],
                kernel_size=self.params.train.kernel_size,
                batch_size=self.params.train.batch_size,
                n_steps_out=output_length,
                output_activation=output_activation,
                classification=self.params.clean.classification,
            )
            # model = nn.bcnn_edward(data_size=X_train.shape[0],
            #                 window_size=X_train.shape[1],
            #                 feature_size=X_train.shape[2],
            #                 kernel_size=self.params.train.kernel_size,
            #                 n_steps_out=output_length,
            #                 output_activation=output_activation,
            #                 classification=self.params.clean.classification)
        else:
            raise NotImplementedError(f"Learning method {self.params.train.learning_method} not implemented.")

        if self.params.train.learning_method in config.NON_DL_METHODS:
            print("Fitting model...")
            model.fit(X_train, y_train)
            print("Done fitting model")
            dump(model, config.MODELS_FILE_PATH)
        else:
            print(model.summary())
            plot_neural_network_architecture(model)

            # if self.params.train.cross_validation:
            #     if self.params.clean.classification:
            #         keras_wrapper = getattr(tf_sklearn, "KerasClassifier")
            #     else:
            #         keras_wrapper = getattr(tf_sklearn, "KerasRegressor")

            #     estimator = keras_wrapper(build_fn=buildmodel, epochs=self.params.train.epochs, batch_size=self.params.train.batch_size, verbose=0)
            #     kfold= RepeatedKFold(n_splits=5, n_repeats=100)
            #     results= cross_val_score(estimator, x, y, cv=kfold, n_jobs=2)  # 2 cpus
            #     results.mean()  # Mean MSE

            if self.params.train.early_stopping:
                early_stopping = EarlyStopping(
                    monitor="val_" + monitor_metric,
                    patience=self.params.train.patience,
                    verbose=4,
                    restore_best_weights=True,
                )

                model_checkpoint = ModelCheckpoint(
                    str(config.MODELS_FILE_PATH), monitor="val_" + monitor_metric  # , save_best_only=True
                )

                # Train model for 10 epochs before adding early stopping
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=10,
                    batch_size=self.params.train.batch_size,
                    validation_split=0.25,
                )

                loss = history.history[monitor_metric]
                val_loss = history.history["val_" + monitor_metric]

                history = model.fit(
                    X_train,
                    y_train,
                    epochs=self.params.train.n_epochs,
                    batch_size=self.params.train.batch_size,
                    validation_split=0.25,
                    callbacks=[early_stopping, model_checkpoint],
                )

                loss += history.history[monitor_metric]
                val_loss += history.history["val_" + monitor_metric]

            else:
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=self.params.train.n_epochs,
                    batch_size=self.params.train.batch_size,
                    validation_split=0.25,
                )

                loss = history.history["loss"]
                val_loss = history.history["val_loss"]

                model.save(config.MODELS_FILE_PATH)


            if self.params.clean.classification:
                best_epoch = np.argmax(np.array(val_loss))
            else:
                best_epoch = np.argmin(np.array(val_loss))

            print(f"Best model in epoch: {best_epoch}")

            n_epochs = range(len(loss))

            plt.figure()
            plt.plot(n_epochs, loss, label="Training loss")
            plt.plot(n_epochs, val_loss, label="Validation loss")
            plt.legend()
            plt.savefig(config.TRAININGLOSS_PLOT_PATH)

def plot_neural_network_architecture(model):
    """Save a plot of the model. Will not work if Graphviz is not installed,
    and is therefore skipped if an error is thrown.

    """
    try:
        plot_model(
            model,
            to_file=config.PLOTS_PATH / "model.png",
            show_shapes=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )
    except:
        print(
            "Failed saving plot of the network architecture, Graphviz must be installed to do that."
        )

def main():
    TrainStage().run()

if __name__ == "__main__":
    main()
