#!/bin/bash
# Quick test runner that automatically activates virtual environment

cd "$(dirname "$0")"
source venv/bin/activate
python test_quant_engine.py
