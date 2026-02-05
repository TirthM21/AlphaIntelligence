#!/bin/bash
# Main screening runner that automatically activates virtual environment

cd "$(dirname "$0")"
source venv/bin/activate
python run_quant_engine.py "$@"
