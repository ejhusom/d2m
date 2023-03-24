#!/bin/bash
# ===================================================================
# File:     build_and_run_udava_docker.sh
# Author:   Erik Johannes Husom
# Created:
# -------------------------------------------------------------------
# Description: Build and run Udava Docker container.
# ===================================================================

docker build -t d2m -f Dockerfile .
docker run -p 5000:5000 -v $(pwd)/assets:/usr/d2m/assets -v $(pwd)/.dvc:/usr/d2m/.dvc d2m
