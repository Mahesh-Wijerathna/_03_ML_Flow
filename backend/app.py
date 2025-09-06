from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Load model from MLflow (latest run in experiment)
experiment_name = "Titanic Survival"
try:
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs.empty:
            latest_run = runs.iloc[0]
            model_uri = f"runs:/{latest_run.run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
        else:
            model = None
    else:
        model = None
except:
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not available'}), 500
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'survived': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)