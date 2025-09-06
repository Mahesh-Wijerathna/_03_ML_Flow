#!/bin/bash

# Wait for MLflow to be ready using Python
echo "Waiting for MLflow..."
python -c "
import time
import requests
while True:
    try:
        requests.get('http://mlflow:5000')
        break
    except:
        time.sleep(2)
"

# Run training
echo "Running training..."
python scripts/train.py

# Run promotion
echo "Running promotion..."
python scripts/promote_model.py

# Start the app
echo "Starting backend..."
python backend/app.py
