#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation of configuration file params.yaml.

Author:
    Erik Johannes Husom

Created:
    2023-11-23 torsdag 14:42:43 

"""
from enum import Enum
from pydantic import BaseModel, validator, Field
from typing import List, Optional, Union

class LearningMethod(str, Enum):
    DNN = 'dnn'
    CNN = 'cnn'
    RNN = 'rnn'
    DT = 'dt'
    RF = 'rf'
    EXPLAINABLEBOOSTING = 'explainableboosting'
    XGBOOST = 'xgboost'
    XGB = 'xgb'
    LGBM = 'lgbm'
    LDA = 'lda'
    QDA = 'qda'
    SVM = 'svm'
    LINEARREGRESSION = 'linearregression'
    RIDGEREGRESSION = 'ridgeregression'
    LASSO = 'lasso'
    LARS = 'lars'
    BAYESIANRIDGE = 'bayesianridge'
    ARDREGRESSION = 'ardregression'
    ELASTICNET = 'elasticnet'
    SGD = 'sgd'
    KNEIGHBORS = 'kneighbors'
    KN = 'kn'
    GB = 'gb'
    GRADIENTBOOSTING = 'gradientboosting'

class ExplanationMethod(str, Enum):
    SHAP = 'shap'
    LIME = 'lime'
    ALL = 'all'

class ProfileParams(BaseModel):
    dataset: str

class CleanParams(BaseModel):
    target: str
    classification: bool
    onehot_encode_target: bool
    combine_files: bool
    percentage_zeros_threshold: float = Field(..., ge=0, le=1)
    correlation_metric: str
    input_max_correlation_threshold: float = Field(..., ge=0)

class FeaturizeParams(BaseModel):
    variables_to_include: Optional[List[str]]
    use_all_engineered_features_on_all_variables: bool
    add_sum: Optional[List[str]]
    add_gradient: Optional[List[str]]
    add_mean: Optional[List[str]]
    add_maximum: Optional[List[str]]
    add_minimum: Optional[List[str]]
    add_min_max_range: Optional[List[str]]
    add_slope: Optional[List[str]]
    add_slope_sin: Optional[List[str]]
    add_slope_cos: Optional[List[str]]
    add_standard_deviation: Optional[List[str]]
    add_variance: Optional[List[str]]
    add_peak_frequency: Optional[List[str]]
    rolling_window_size_sum: int
    rolling_window_size_mean: int
    rolling_window_size_max_min: int
    rolling_window_size_standard_deviation: int
    remove_features: Optional[List[str]]
    target_min_correlation_threshold: float

class SplitParams(BaseModel):
    train_split: float = Field(..., ge=0, le=1)
    shuffle_files: bool
    shuffle_samples_before_split: bool

class ScaleParams(BaseModel):
    input_method: str
    output_method: Optional[str]

class SequentializeParams(BaseModel):
    window_size: int
    overlap: int
    target_size: int
    shuffle_samples: bool
    future_predict: bool

class TrainParams(BaseModel):
    seed: int
    learning_method: LearningMethod
    ensemble: bool
    hyperparameter_tuning: bool
    n_epochs: int
    early_stopping: bool
    patience: int
    activation_function: str
    batch_size: int
    n_layers: int
    n_neurons: Union[int, List[int]]
    dropout: float = Field(..., ge=0, le=1)
    n_flattened_layers: int
    n_flattened_nodes: int
    kernel_size: int
    maxpooling: bool
    maxpooling_size: int
    unit_type: str
    ff_dim: int
    n_transformer_blocks: int
    n_heads: int
    head_size: int

class EvaluateParams(BaseModel):
    performance_metric: str
    threshold_for_ensemble_models: float
    dropout_uncertainty_estimation: bool
    uncertainty_estimation_sampling_size: int
    show_inputs: bool

class ExplainParams(BaseModel):
    generate_explanations: bool
    number_of_background_samples: int
    number_of_summary_samples: int
    explanation_method: ExplanationMethod
    seed: int

class CombineExplanationsParams(BaseModel):
    combination_method: str
    weighting_method: str
    agreement_method: str

class AllParams(BaseModel):
    profile: ProfileParams
    clean: CleanParams
    featurize: FeaturizeParams
    split: SplitParams
    scale: ScaleParams
    sequentialize: SequentializeParams
    train: TrainParams
    evaluate: EvaluateParams
    explain: ExplainParams
    combine_explanations: CombineExplanationsParams

