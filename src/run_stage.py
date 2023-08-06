#!/usr/bin/env python3
"""Run a pipeline stage.

Author:
    Erik Johannes Husom

Created:
    2023-08-06

    """
import sys

from codecarbon import track_emissions

from profiling import ProfileStage

def run_stage(stage_name):
    
    if stage_name == "profile":
        ProfileStage().run()

if __name__ == "__main__":
    
    stage_name = sys.argv[1]
    run_stage(stage_name)