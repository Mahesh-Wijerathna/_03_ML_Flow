import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Set MLflow tracking URI
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load data (assume train.csv is in data/)
data_path = os.getenv("DATA_PATH", "../data/train.csv")  # Use env var for flexibility
data = pd.read_csv(data_path)

# Simple preprocessing (adjust as needed)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()

X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and log
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# MLflow logging
mlflow.set_experiment("Titanic Survival")
with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

print(f"Model trained with accuracy: {accuracy}")