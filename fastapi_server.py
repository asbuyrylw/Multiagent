
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from agents.orchestrator_agent import OrchestratorAgent
import shutil
import os

app = FastAPI(title="Multi-Agent API", version="1.0")

@app.post("/run-workflow")
async def run_workflow(
    workflow_name: str = Form(...),
    training_file: UploadFile = None,
    prediction_file: UploadFile = None
):
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    if training_file:
        train_path = os.path.join(data_dir, "churn_training.csv")
        with open(train_path, "wb") as f:
            shutil.copyfileobj(training_file.file, f)

    if prediction_file:
        pred_path = os.path.join(data_dir, "churn_to_predict.csv")
        with open(pred_path, "wb") as f:
            shutil.copyfileobj(prediction_file.file, f)

    workflow_path = f"workflows/{workflow_name}.yaml"
    orchestrator = OrchestratorAgent(workflow_path)
    orchestrator.run()
    return {"status": "completed", "workflow": workflow_name}
