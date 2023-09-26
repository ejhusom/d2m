#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile dataset.

Created:
    2021-06-30

"""
import sys

import pandas as pd
import yaml
from codecarbon import track_emissions
from ydata_profiling import ProfileReport

from config import config
from preprocess_utils import find_files
from pipelinestage import PipelineStage

@track_emissions(project_name="profiling")
class ProfileStage(PipelineStage):
    
    def __init__(self):
        super().__init__(stage_name="profile")
        
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
    
    
        # Save report to files.
        profile.to_file(config.PROFILE_JSON_PATH)
        profile.to_file(config.PROFILE_HTML_PATH)

if __name__ == "__main__":

    ProfileStage().run()
