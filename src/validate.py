#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation of configuration file params.yaml.

Author:
    Erik Johannes Husom

Created:
    2023-08-05 Saturday 23:26:26 

"""
from config import *

class BaseConfig:
    def validate(self):
        pass

class ProfileConfig:
    def __init__(self, dataset):
        self.dataset = dataset

class CleanConfig:
    def __init__(self, target, classification, onehot_encode_target, combine_files, percentage_zeros_threshold, correlation_metric, input_max_correlation_threshold):
        self.target = target
        self.classification = classification
        self.onehot_encode_target = onehot_encode_target
        self.combine_files = combine_files
        self.percentage_zeros_threshold = percentage_zeros_threshold
        self.correlation_metric = correlation_metric
        self.input_max_correlation_threshold = input_max_correlation_threshold

class FeaturizeConfig:
    def __init__(self, 
                 variables_to_include, 
                 use_all_engineered_features_on_all_variables,
                 add_sum, add_gradient, add_mean, add_maximum, add_minimum, 
                 add_min_max_range, add_slope, add_slope_sin, add_slope_cos, 
                 add_standard_deviation, add_variance, add_peak_frequency, 
                 rolling_window_size_sum, rolling_window_size_mean,
                 rolling_window_size_max_min, rolling_window_size_standard_deviation,
                 remove_features, target_min_correlation_threshold):
        
        self.variables_to_include = variables_to_include
        self.use_all_engineered_features_on_all_variables = use_all_engineered_features_on_all_variables
        
        # Feature Engineering Methods
        self.add_sum = add_sum
        self.add_gradient = add_gradient
        self.add_mean = add_mean
        self.add_maximum = add_maximum
        self.add_minimum = add_minimum
        self.add_min_max_range = add_min_max_range
        self.add_slope = add_slope
        self.add_slope_sin = add_slope_sin
        self.add_slope_cos = add_slope_cos
        self.add_standard_deviation = add_standard_deviation
        self.add_variance = add_variance
        self.add_peak_frequency = add_peak_frequency
        
        # Rolling Window Sizes
        self.rolling_window_size_sum = rolling_window_size_sum
        self.rolling_window_size_mean = rolling_window_size_mean
        self.rolling_window_size_max_min = rolling_window_size_max_min
        self.rolling_window_size_standard_deviation = rolling_window_size_standard_deviation
        
        # Features to Remove after Engineering
        self.remove_features = remove_features
        
        # Correlation Threshold
        self.target_min_correlation_threshold = target_min_correlation_threshold

    def validate(self):
        # Insert validation logic here
        # For instance, check if the lengths of lists match where required,
        # or if certain values fall within acceptable ranges.
        pass

class SplitConfig:
    def __init__(self, train_split, shuffle_files, shuffle_samples_before_split):
        self.train_split = train_split
        self.shuffle_files = shuffle_files
        self.shuffle_samples_before_split = shuffle_samples_before_split
class ScaleConfig:
    def __init__(self, input, output):
        self.input = input
        self.output = output
class SequentializeConfig:
    def __init__(self, window_size, overlap, target_size, shuffle_samples, future_predict):
        self.window_size = window_size
        self.overlap = overlap
        self.target_size = target_size
        self.shuffle_samples = shuffle_samples
        self.future_predict = future_predict
class TrainConfig:
    def __init__(self, seed, learning_method, ensemble, hyperparameter_tuning,
                 n_epochs, early_stopping, patience, activation_function, batch_size,
                 n_layers, n_neurons, dropout, n_flattened_layers, n_flattened_nodes,
                 kernel_size, maxpooling, maxpooling_size, unit_type, ff_dim,
                 n_transformer_blocks, n_heads, head_size):

        # General parameters
        self.seed = seed
        self.learning_method = learning_method
        self.ensemble = ensemble
        self.hyperparameter_tuning = hyperparameter_tuning

        # Parameters for deep learning (dnn, cnn, lstm, etc.)
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.dropout = dropout

        # Parameters for flattened layers (relevant for cnn and rnn)
        self.n_flattened_layers = n_flattened_layers
        self.n_flattened_nodes = n_flattened_nodes

        # Parameters for convolutional neural network (cnn)
        self.kernel_size = kernel_size
        self.maxpooling = maxpooling
        self.maxpooling_size = maxpooling_size

        # Parameters for recurrent neural network (rnn)
        self.unit_type = unit_type

        # Parameters for transformers
        self.ff_dim = ff_dim
        self.n_transformer_blocks = n_transformer_blocks
        self.n_heads = n_heads
        self.head_size = head_size

    def validate(self):
        # Insert validation logic here. For example:

        if self.learning_method not in ["dnn", "cnn", "lstm", "transformer"]:
            raise ValueError(f"Unsupported learning method: {self.learning_method}")

        if self.early_stopping and not isinstance(self.patience, int):
            raise ValueError("If early stopping is enabled, patience should be an integer.")

        # Add more validation checks as needed based on your constraints and requirements.

class EvaluateConfig:
    def __init__(self, performance_metric, threshold_for_ensemble_models, dropout_uncertainty_estimation, uncertainty_estimation_sampling_size, show_inputs):
        self.performance_metric = performance_metric
        self.threshold_for_ensemble_models = threshold_for_ensemble_models
        self.dropout_uncertainty_estimation = dropout_uncertainty_estimation
        self.uncertainty_estimation_sampling_size = uncertainty_estimation_sampling_size
        self.show_inputs = show_inputs
class ExplainConfig:
    def __init__(self, number_of_background_samples, number_of_summary_samples, explanation_method, seed):
        self.number_of_background_samples = number_of_background_samples
        self.number_of_summary_samples = number_of_summary_samples
        self.explanation_method = explanation_method
        self.seed = seed

class MainConfig:
    def __init__(self, **configs):
        self.profile = ProfileConfig(configs.get("profile"))
        self.clean = CleanConfig(**configs.get("clean"))
        self.featurize = FeaturizeConfig(**configs.get("featurize"))
        self.split = SplitConfig(**configs.get("split")) self.scale = ScaleConfig(**configs.get("scale"))
        self.sequentialize = SequentializeConfig(**configs.get("sequentialize"))
        self.train = TrainConfig(**configs.get("train"))
        self.evaluate = EvaluateConfig(**configs.get("evaluate"))
        self.explain = ExplainConfig(**configs.get("explain"))

    def validate(self):
        # Here, you can call the validate method of each sub-configuration.
        self.clean.validate()
        self.featurize.validate()
        self.split.validate()
        self.scale.validate()
        self.sequentialize.validate()
        self.train.validate()
        self.evaluate.validate()
        self.explain.validate()

    # def create_folders(self):


        

if __name__ == '__main__':

    data = yaml.safe_load(open('params.yaml'))
    main_config = MainConfig(**data)
    main_config.validate()
