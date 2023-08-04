#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for creating reports and tables.

Author:
    Erik Johannes Husom

Created:
    2023-08-02 onsdag 11:40:51 

"""
from config import FEATURES_PATH

def generate_ensemble_explanation_tables(
        sorted_feature_importances,
        adequate_methods,
        n_tables = 2,
        n_most_important_features = 10,
    ):

    n_methods_per_table = len(adequate_methods) // n_tables
    overflow_methods = len(adequate_methods) % n_tables
    
    for t in range(n_tables):
        first_method = t * n_methods_per_table

        if overflow_methods > 0 and t == n_tables - 1:
            last_method = (t+1) * n_methods_per_table + overflow_methods
        else:
            last_method = (t+1) * n_methods_per_table

        header_row = ""

        # Create header row for importance_table.
        for method in adequate_methods[first_method:last_method]:
            header_row += f" & {method} " + r"Feature & $\bar{S}$ "

        rows = [header_row]

        for i in range(n_most_important_features):
            row = f"{i+1} "
            for method in adequate_methods[first_method:last_method]:
                df = sorted_feature_importances[method]

                feature = df.index[i].replace("_", r"\_")
                value = df[f"feature_importance_{method}"][i]
                row += r" & \texttt{" + feature + "}" + f" & {round(value, 3)} "

            rows.append(row)

        
        importance_table = "\\\\ \n".join(rows) + "\\\\"

        with open(FEATURES_PATH / f"importance_table_{t}.tex", "w") as f:
            f.write(importance_table)
