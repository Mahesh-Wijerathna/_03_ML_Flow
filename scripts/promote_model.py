import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

# Set MLflow tracking URI (local)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

client = MlflowClient()

# Get the latest run from the experiment
experiment = mlflow.get_experiment_by_name("Titanic Survival")
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs.empty:
        latest_run = runs.iloc[0]
        latest_accuracy = latest_run["metrics.accuracy"]
        
        # Ensure the registered model exists
        try:
            client.get_registered_model("TitanicClassifier")
        except:
            client.create_registered_model("TitanicClassifier")
        
        # Get current production model version
        try:
            prod_version = client.get_latest_versions("TitanicClassifier", stages=["Production"])[0]
            prod_accuracy = mlflow.get_run(prod_version.run_id).data.metrics.get("accuracy", 0)
            
            if latest_accuracy > prod_accuracy:
                # Promote to production
                client.transition_model_version_stage(
                    name="TitanicClassifier",
                    version=prod_version.version,
                    stage="Archived"
                )
                new_version = client.create_model_version(
                    name="TitanicClassifier",
                    source=f"runs:/{latest_run.run_id}/model",
                    run_id=latest_run.run_id
                )
                client.transition_model_version_stage(
                    name="TitanicClassifier",
                    version=new_version.version,
                    stage="Production"
                )
                print("Model promoted to Production")
            else:
                print("New model not better; not promoted")
        except:
            # If no production model, create one
            new_version = client.create_model_version(
                name="TitanicClassifier",
                source=f"runs:/{latest_run.run_id}/model",
                run_id=latest_run.run_id
            )
            client.transition_model_version_stage(
                name="TitanicClassifier",
                version=new_version.version,
                stage="Production"
            )
            print("Initial model set to Production")
    else:
        print("No runs found")
else:
    print("Experiment not found")
