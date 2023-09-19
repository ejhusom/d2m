#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combine explanations.

Author:
    Erik Johannes Husom

Created:
    2023-09-08 fredag 14:26:05 

"""
import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr

from pipelinestage import PipelineStage
from config import config

class CombineExplanationsStage(PipelineStage):
    def __init__(self):
        super().__init__(stage_name="combine_explanations")

    def run(self):

        feature_importances = pd.read_csv(config.FEATURES_PATH /
                "feature_importances_adequate_models.csv", index_col=0)

        combine_ensemble_explanations(feature_importances,
                method=self.params.combine_explanations.combination_method,
                weighting_method=self.params.combine_explanations.weighting_method
                )

        agreement_matrix, agreement_score, agreement_std = evaluate_explanation_agreement(feature_importances,
                method=self.params.combine_explanations.agreement_method)

        print(agreement_score)
        print(agreement_std)

        generate_explanation_report()

def combine_ensemble_explanations(feature_importances, method="avg",
        weighting_method="softmax"):
    """Combine explanations from ensemble.

    Args:
        feature_importances: DatFrame containing the feature importances for
            all models in ensemble. Rows: Models. Columns: Features.

    """

    # Drop the models with invalid (nan) feature importances.
    feature_importances = feature_importances.dropna()

    if method in ["avg", "average", "mean"]:
        combined_feature_importances = feature_importances.mean(0, numeric_only=True)
    elif method in ["weighted", "weight", "weights"]:

        # Load adequate models and their scores
        with open(config.ADEQUATE_MODELS_FILE_PATH, "r") as f:
            adequate_models = json.load(f)

        model_scores = pd.DataFrame.from_dict(adequate_models, orient="index")

        if weighting_method == "normalize":
            # Normalize the scores so they sum to 1.  Useful When: You want all
            # model scores to be adjusted to a consistent scale without making
            # any distinction between positive and negative scores. It retains
            # the relative importance of scores.  Less Useful: If you have
            # scores with large variances or outliers, as they might dominate
            # the normalized scores. Also, negative scores are still included,
            # which might not be desirable if negative values lack meaningful
            # interpretation.
            weights = model_scores / model_scores.sum()
        elif weighting_method == "softmax":
            # Apply the softmax function to the scores.     Useful When: You
            # want to emphasize differences between scores while ensuring
            # they're in the range [0, 1] and sum to 1. Higher scores will be
            # amplified more than lower ones, turning scores into a kind of
            # "probability distribution." Less Useful: When scores are very
            # close in magnitude, as differences might be overly exaggerated.
            weights = pd.DataFrame(
                    np.exp(model_scores.iloc[:, 0]) /
                    np.exp(model_scores.iloc[:, 0]).sum())
        elif weighting_method == "clip_negative":
            # Clip negative values to 0, and then normalize the scores so they sum to 1.
            # Useful When: Negative scores are considered non-informative
            # or misleading, and you'd rather treat them as a baseline. By
            # setting them to zero, you neutralize their effect, and by
            # normalizing afterward, you adjust the remaining scores to a
            # consistent scale.  Less Useful: When negative scores have a
            # specific meaningful interpretation, as this method disregards
            # that information.
            model_scores.iloc[:, 0] = model_scores.iloc[:, 0].clip(lower=0)
            weights = model_scores / model_scores.sum()
        elif weighting_method == "square":
            # Take the square of each score. This emphasizes higher scores and
            # further diminishes the impact of lower scores, even if they're
            # positive.     Useful When: You want to emphasize models with
            # higher scores and diminish the impact of lower scores. Squaring
            # exaggerates differences and can help in highlighting
            # top-performing models.  Less Useful: When the scores are close in
            # magnitude or when small differences in scores are important, as
            # this method might overshadow them.
            squared_weights = model_scores.iloc[:, 0] ** 2
            weights = squared_weights / squared_weights.sum()
        elif weighting_method == "rank":
            # Scale scores based on their rank. This is a non-parametric method
            # and can be useful of you suspect outliers or extreme values in
            # your scores.
            # Useful When: You are concerned about outliers or extreme
            # values skewing the weights. This method is non-parametric and
            # focuses on the relative rank rather than the exact score
            # values.  Less Useful: If the actual magnitude of the scores
            # is important for interpretation, as this method disregards
            # the exact values.
            rank_weights = model_scores.iloc[:, 0].rank()
            weights = rank_weights / rank_weights.sum()
        elif weighting_method == "zscore":
            # Useful When: You're interested in understanding how each
            # model's performance deviates from the average performance. This
            # gives a measure of how many standard deviations a score is from
            # the mean.  Less Useful: When the overall distribution of scores
            # isn't approximately normal, as the interpretation of z-scores
            # is most meaningful in the context of normally distributed data.
            mean_val = model_scores.iloc[:, 0].mean()
            std_val = model_scores.iloc[:, 0].std()
            weights = (model_scores.iloc[:, 0] - mean_val) / std_val
        elif weighting_method == "log":
            # Useful When: There's a vast range in scores, and you want to
            # compress this range to make differences more interpretable.
            # Especially useful when scores span several orders of
            # magnitude.  Less Useful: For negative scores or scores very
            # close to zero, as they can result in large negative or
            # near-infinite values.
            offset = abs(model_scores.iloc[:, 0].min()) + 1
            log_weights = np.log(model_scores.iloc[:, 0] + offset)
            weights = log_weights / log_weights.sum()
        elif weighting_method == "exponential":
            # Useful When: You want to heavily prioritize models with higher
            # scores. The exponential transformation will significantly amplify
            # differences between scores.  Less Useful: When scores are already
            # large, this could lead to extremely large weights, and minor
            # differences in scores could result in exaggerated weight
            # differences.
            exp_weights = np.exp(model_scores.iloc[:, 0])
            weights = exp_weights / exp_weights.sum()
        elif weighting_method == "minmax":
            # Useful When: You want to transform scores to a specific range
            # (e.g., between 0 and 1) while retaining their relative
            # differences.  Less Useful: When there are outliers in the
            # scores, as they can compress most scores into a narrow range,
            # making differences between them less apparent.
            min_val = model_scores.iloc[:, 0].min()
            max_val = model_scores.iloc[:, 0].max()
            weights = (model_scores.iloc[:, 0] - min_val) / (max_val - min_val)
        else:
            # Normalize the scores so they sum to 1
            weights = model_scores / model_scores.sum()

        weights = pd.DataFrame(weights)

        before = feature_importances.mean(0, numeric_only=True)

        for model_name, weight in weights.iterrows():
            model_type = os.path.splitext(model_name)[0].split("_")[-1]

            for index, row in feature_importances.iterrows():
                current_model_type = index.split("_")[-1]
                if model_type == current_model_type:
                    feature_importances.loc[index] = row * weight.iloc[0]

        combined_feature_importances = feature_importances.mean(0, numeric_only=True)

        # Normalize the weighted feature importances in order to make them sum
        # to 1, and be comparable to other results.
        combined_feature_importances = combined_feature_importances / combined_feature_importances.sum()

    else:
        print("Using default combination method: average")
        combined_feature_importances = feature_importances.mean(0, numeric_only=True)

    sorted_combined_feature_importances = combined_feature_importances.sort_values(
        ascending=False
    )
    sorted_combined_feature_importances.to_csv(
        config.FEATURES_PATH / "sorted_combined_feature_importances.csv"
    )

    return sorted_combined_feature_importances

def evaluate_explanation_agreement(feature_importances, method="spearman"):

    # Drop the models with invalid (nan) feature importances.
    feature_importances = feature_importances.dropna()

    model_names = feature_importances.index
    num_models = len(feature_importances)
    correlation_matrix = np.zeros((num_models, num_models))

    if method == "pearson":
        for i in range(num_models):
            for j in range(num_models):
                correlation, _ = pearsonr(
                        feature_importances.iloc[i],
                        feature_importances.iloc[j]
                )
                correlation_matrix[i, j] = correlation
    elif method == "spearman":
        for i in range(num_models):
            for j in range(num_models):
                correlation, _ = spearmanr(
                        feature_importances.iloc[i],
                        feature_importances.iloc[j]
                )
                correlation_matrix[i, j] = correlation

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, square=True,
                xticklabels=model_names, yticklabels=model_names, vmin=-1,
                vmax=1)
    plt.title(f"Pairwise {method} correlation between models")
    plt.savefig(config.PLOTS_PATH / "agreement_matrix.png")

    # Extract lower triangle of correlation matrix, i.e., the relevant values
    # for computing the average agreement.
    relevant_values = []

    for i in range(num_models):
        for j in range(num_models):
            if j < i:
                relevant_values.append(correlation_matrix[i, j])

    agreement = np.mean(relevant_values)
    agreement_std = np.std(relevant_values)

    return correlation_matrix, agreement, agreement_std

def generate_explanation_report():
    with open(config.PLOTS_PATH / "prediction.html", "r") as infile:
        prediction_plot = infile.read()

    with open(config.PLOTS_PATH / "feature_importances.html", "r") as infile:
        feature_importances_plot = infile.read()

    sorted_combined_feature_importances_filepath = (
        config.FEATURES_PATH / "sorted_combined_feature_importances.csv"
    )
    sorted_combined_feature_importances_table = generate_html_table(
        sorted_combined_feature_importances_filepath
    )

    html = "<html>\n"
    html += "<head>\n"
    html += "<meta charset='UTF-8'>"
    html += "<title>Model prediction and explanations</title>"
    html += "<link href='../../src/static/style.css' rel='stylesheet' type='text/css' title='Stylesheet'>"
    html += "<link rel='icon' type='image/png' href='../../src/static/favicon.png'>"
    html += "<script src='../../src/static/jquery.js'></script>"
    html += "</head>"

    html += "<body>"
    html += "<header>"
    html += "<div id=logoContainer>"
    html += "<img src='../../src/static/sintef-logo-centered-negative.svg' id=logo>"
    html += "<h1>Model prediction and explanations</h1>"
    html += "</div>"
    html += "<nav>"
    html += "    <a href='#prediction'>True vs predicted values</a>"
    html += "    <a href='#featureimportanceschart'>Feature importances chart</a>"
    html += "    <a href='#featureimportancestable'>Feature importances table</a>"
    html += "</nav>"
    html += "</header>"

    html += "<div class=box>"
    html += "<h2 id='featureimportancestable'>Feature importances table</h2>"
    html += "<div class=overviewTable>"
    html += sorted_combined_feature_importances_table
    html += "</div>"
    html += "</div>"

    html += "<div class=box>"
    html += "<h2 id='prediction'>True vs predicted values</h2>"
    html += prediction_plot
    html += "</div>"

    html += "<div class=box>"
    html += "<h2 id='featureimportanceschart'>Feature importances chart</h2>"
    html += feature_importances_plot
    html += "</div>"

    html += "</body>"
    html += "</html>"

    with open("assets/plots/report.html", "w") as outfile:
        outfile.write(html)


def generate_html_table(csv_file):
    table_html = "<table>\n"
    with open(csv_file, "r") as file:
        csv_reader = csv.reader(file)
        header_row = next(csv_reader)
        table_html += "  <thead>\n"
        table_html += "    <tr>\n"
        table_html += f"      <th>Feature name</th>\n"
        table_html += f"      <th>Impact score</th>\n"
        table_html += "    </tr>\n"
        table_html += "  </thead>\n"

        table_html += "  <tbody>\n"
        for row in csv_reader:
            table_html += "    <tr>\n"
            for i, cell in enumerate(row):
                if i == 1:  # Format numbers in the second column
                    formatted_cell = "{:.3f}".format(float(cell))
                else:
                    formatted_cell = cell
                table_html += f"      <td>{formatted_cell}</td>\n"
            table_html += "    </tr>\n"
        table_html += "  </tbody>\n"
    table_html += "</table>"
    return table_html


def main():
    CombineExplanationsStage().run()

if __name__ == "__main__":
    main()
