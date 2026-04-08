#!/bin/bash
sed -e 's/provider: mock/provider: google/g' -e 's/model_name: mock-v1/model_name: gemini-2.5-flash/g' experiments/health_insurance_choice.yaml > test_health_insurance.yaml
export GEMINI_API_KEY="AIzaSyTestKeyForMocking"
PYTHONPATH=src .venv/bin/python -m tool_lab.cli run --config test_health_insurance.yaml
