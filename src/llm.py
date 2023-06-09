#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for producing natural language explanation and reasoning about the
model.

Author:
    Erik Johannes Husom

Created:
    2023-04-19 onsdag 13:26:49 

"""
import yaml
import pandas as pd
import numpy as np
import openai

from config import *

def llm():

    # Load parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    dataset = params["profile"]["dataset"]
    target = params["clean"]["target"]
    window_size = params["sequentialize"]["window_size"]
    overlap = params["sequentialize"]["overlap"]
    learning_method = params["train"]["learning_method"]
    dataset_description = params["llm"]["dataset_description"]
    
    # Load SHAP importances
    shap_importances = pd.read_csv(OUTPUT_PATH / "shap_importances.csv")
    print(shap_importances)

    # Generate prompt for LLM model, based on the parameters
    user_message = f"""\
The dataset is called {dataset}.
The target is {target}.
The description of the dataset is as follows: {dataset_description}.
The window size is {window_size}.
The overlap is {overlap}.
The learning method is {learning_method}.
The 5 most important features are {shap_importances["feature"][0:5].tolist()}.
"""

    # user_message += f"""\
# Can you explain the model to me? Additionally, can you
# explain why the model makes the predictions it makes? Can you also explain
# why the most important features have influence on the target variable {target}?"""
    user_message += f"""\
Why are the most important features important for the target variable {target}?
Can you reason about why exactly those features have influence on the target variable {target}?"""

    system_message = f"""\
You are an AI assistant that is trying to help a user understand the output
from a machine learning (ML) pipeline. The ML pipeline analyzes a dataset
consisting of time series or tabular data. The ML pipeline produces a
prediction based on the input data. The prediction is a number or a
classification. The ML pipeline also produces a feature importance ranking
for the input data. The feature importance ranking is a list of the input
features, sorted by how important they are for the predictions. Your role is to
produce a natural language explanation and reasoning about the ML model and the
most important features.
"""
# The
# explanation and reasoning should be understandable by a non-technical user. The
# target value is what the ML model is trying to predict.

    print(user_message)
    print(system_message)


    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            # {"role": "user", "content": "Where was it played?"}
        ]
    )

    print(response)
    result = response['choices'][0]['message']['content']

    # Save result in a text file
    with open(OUTPUT_PATH / "llm.txt", "w") as f:
        f.write(result)

if __name__ == '__main__':
    llm()
