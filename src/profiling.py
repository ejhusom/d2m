#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile dataset.

Created:
    2021-06-30

"""
import sys

import pandas as pd
import json
import yaml
from codecarbon import track_emissions
from ydata_profiling import ProfileReport

from config import config
from preprocess_utils import find_files
from pipelinestage import PipelineStage

class ProfileStage(PipelineStage):
    
    def __init__(self):
        super().__init__(stage_name="profile")
        
    @track_emissions(project_name="profiling")
    def run(self):
        """Creates a profile report of a data set.

        Reads data from a set of input files, and creates a report containing
        profiling of the data. This profiling consists of various statistical
        properties. The report is stored in two formats:

        - HTML: For visual inspection
        - JSON: For subsequent automatic processing of results

        """

        filepaths = find_files(self.raw_data_path, file_extension=".csv")
    
        dfs = []
    
        for filepath in filepaths:
            dfs.append(pd.read_csv(filepath))
    
        combined_df = pd.concat(dfs, ignore_index=True)
    
        # Generate report.
        profile = ProfileReport(
            combined_df,
            title="Profiling Analysis",
            config_file="src/profile.yaml",
            lazy=False,
            sort=None,
        )

        # Save report to HTML
        profile.to_file(config.PROFILE_HTML_PATH)

        # Delete unnecessary information to save disk space
        profile_json = json.loads(profile.json)

        for variable in profile_json["variables"].keys():
            del profile_json["variables"][variable]["value_counts_without_nan"]
            del profile_json["variables"][variable]["value_counts_index_sorted"]

        with open(config.PROFILE_JSON_PATH, "w") as f:
            json.dump(profile_json, f)

if __name__ == "__main__":

    ProfileStage().run()
