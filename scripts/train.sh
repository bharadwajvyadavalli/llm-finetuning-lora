#!/usr/bin/env bash
set -e
python src/main.py --mode preprocess --config config/default.yaml
python src/main.py --mode train --config config/default.yaml
