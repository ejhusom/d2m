#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualizing carbon footprint of experiments.

Author:
    Erik Johannes Husom

Created:
    2023-04-14 fredag 15:15:13 

"""
import logging
import re
import subprocess

import codecarbon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from config import config

logging.basicConfig(
        filename='d2m.log', 
        filemode='a', 
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        level=logging.INFO,
)

def read_emissions_file(filepath=config.EMISSIONS_FILE_PATH):
    """Read emissions file.

    Args:
        filepath (str): Path to emissions file.

    Returns:
        DataFrame: Emissions data.

    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    return df

def plot_emissions(df, title, filepath):
    """Plot emissions.

    Args:
        df (DataFrame): Emissions data.
        title (str): Title of plot.
        filepath (str): Path to save plot.

    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["emissions"])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Emissions (gCO2e)")
    plt.savefig(filepath)
    plt.close()

def run_pipeline():
    """Run DVC pipeline."""

    result = subprocess.run(["dvc", "repro"], check=True, capture_output=True)
    dvc_output = result.stdout.decode("utf-8")
    print(dvc_output)
    
    # define the regex pattern for finding skipped stages
    skipped_stages_pattern = r"Stage '(.*?)' didn't change, skipping"

    # find all matches using the regex pattern
    skipped_stages = re.findall(skipped_stages_pattern, dvc_output)

    emissions_file_headers = list(codecarbon.output.EmissionsData.__annotations__.keys())
    print(emissions_file_headers)

    for stage in skipped_stages:
        logging.info(f"Stage {stage} was skipped.")


def plot_average_emissions_per_stage(
        avg_emissions_per_stage,
        std_emissions_per_stage,
        metrics=["duration", "emissions", "energy_consumed"]
    ):

    average_values = {metric: [] for metric in metrics}
    std_dev_values = {metric: [] for metric in metrics}

    for metric in metrics:
        average_values[metric] = avg_emissions_per_stage[metric]
        std_dev_values[metric] = std_emissions_per_stage[metric]

    traces = []

    for metric in metrics:
        trace = go.Bar(
            x=stages,
            y=average_values[metric],
            name=f"Average {metric}",
            error_y=dict(type='data', array=std_dev_values[metric], visible=True),
        )
        traces.append(trace)

    layout = go.Layout(
        title="Pipeline Stage Metrics",
        xaxis=dict(title="Pipeline Stage"),
        yaxis=dict(title="Value"),
        barmode="group",
    )

    fig = go.Figure(data=traces, layout=layout)

    fig.show()


def main():
    """Main function.

    Useful information:

    - Total emissions
    - Total emissions per stage
    - Avg + std emissions per stage
    - Avg + std emissions for full pipeline run


    """

    # Read emissions data
    df = read_emissions_file(config.EMISSIONS_FILE_PATH)
    stages = df["project_name"].unique()
    df["duration_min"] = df["duration"] / 60

    # total_emissions = df.sum()

    total_emissions_per_stage = df.groupby(["project_name"]).sum(numeric_only=True)

    avg_emissions_per_stage = df.groupby(["project_name"]).mean(numeric_only=True)
    std_emissions_per_stage = df.groupby(["project_name"]).std(numeric_only=True)

    avg_emissions_full_pipeline = avg_emissions_per_stage.sum(numeric_only=True)
    std_emissions_full_pipeline = np.sqrt(df.groupby(["project_name"]).var(numeric_only=True).sum(numeric_only=True))

    plot_average_emissions_per_stage(avg_emissions_per_stage, std_emissions_per_stage, metrics = ["energy_consumed"])

    # Plot emissions
    # plot_emissions(df, "Emissions", "emissions.png")

if __name__ == "__main__":
    main()
    # run_pipeline()
