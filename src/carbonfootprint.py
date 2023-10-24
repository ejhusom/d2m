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

logging.basicConfig(
        filename='d2m.log', 
        filemode='a', 
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        level=logging.INFO,
)

def read_emissions_file(filepath):
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


def main():
    """Main function."""

    # Read emissions data
    emissions = read_emissions_file("emissions.csv")

    # Plot emissions
    plot_emissions(emissions, "Emissions", "emissions.png")

if __name__ == "__main__":
    main()
    # run_pipeline()

