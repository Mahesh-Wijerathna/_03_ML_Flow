import subprocess
import time

# Step 1: Start MLflow UI (local)
print("Starting MLflow UI...")
subprocess.Popen(['mlflow', 'ui', '--host', '0.0.0.0', '--port', '5000'])

# Wait a bit
time.sleep(5)

# Step 2: Run training
print("Running training...")
subprocess.run(['python', 'scripts/train.py'])

# Step 3: Promote model
print("Promoting model...")
subprocess.run(['python', 'scripts/promote_model.py'])

# Step 4: Start backend
print("Starting backend...")
subprocess.Popen(['python', 'backend/app.py'])

# Step 5: Start frontend (assuming Node.js is installed)
print("Starting frontend...")
subprocess.Popen(['node', 'frontend/app.js'])

print("Pipeline complete. Access:")
print("- MLflow UI: http://localhost:5000")
print("- Backend API: http://localhost:5001/predict")
print("- Frontend: http://localhost:3000")
