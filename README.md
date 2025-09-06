# Titanic Survival Predictor MLOps Project

This project demonstrates an end-to-end MLOps pipeline using MLflow for a Titanic survival prediction model.

## Structure
- `data/`: Dataset (train.csv)
- `scripts/`: Training script (train.py), promotion script (promote_model.py)
- `backend/`: Flask API (app.py)
- `frontend/`: Node.js frontend (app.js, index.html)
- `run_pipeline.py`: Local pipeline runner
- `docker-compose.yml`: Docker Compose for containerized deployment

## Local Setup
1. Install dependencies: `pip install -r requirements.txt`
2. For frontend: `cd frontend && npm init -y && npm install express axios`
3. Add Titanic dataset to `data/train.csv`
4. Run the full pipeline: `python run_pipeline.py`
5. Access:
   - MLflow UI: http://localhost:5000
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5001/predict

## Docker Setup
1. Ensure Docker and Docker Compose are installed.
2. Add Titanic dataset to `data/train.csv`
3. Run: `docker-compose up --build`
4. Access the same URLs as above.

## CI/CD with GitHub Actions and Railway
1. Push code to `main` branch.
2. GitHub Actions will:
   - Train and test the model.
   - Deploy to Railway.
3. Railway Setup:
   - Create a Railway project.
   - Connect to your GitHub repo.
   - Set environment variables: `MLFLOW_TRACKING_URI`, etc.
   - Add secrets: `RAILWAY_TOKEN`, `RAILWAY_PROJECT_ID` in GitHub repo settings.

## MLflow
- Start MLflow UI: `mlflow ui`
- Register model in registry as 'TitanicClassifier'