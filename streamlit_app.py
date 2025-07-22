
import streamlit as st
import yaml
import os
import pandas as pd
from agents.orchestrator_agent import OrchestratorAgent

st.set_page_config(page_title="Multi-Agent System UI", layout="wide")

st.title("🧠 Multi-Agent Workflow Runner")

# Select workflow
workflow_dir = "workflows"
workflows = [f for f in os.listdir(workflow_dir) if f.endswith(".yaml")]
selected_workflow = st.selectbox("Choose a Workflow", workflows)

# Upload CSVs if needed
st.subheader("Upload Data Files for ML")
training_file = st.file_uploader("Training Data (CSV)", type="csv", key="train")
predict_file = st.file_uploader("Prediction Data (CSV)", type="csv", key="predict")

# Save files if uploaded
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
if training_file:
    train_path = os.path.join(data_dir, "churn_training.csv")
    with open(train_path, "wb") as f:
        f.write(training_file.getvalue())
    st.success("Training data uploaded.")

if predict_file:
    pred_path = os.path.join(data_dir, "churn_to_predict.csv")
    with open(pred_path, "wb") as f:
        f.write(predict_file.getvalue())
    st.success("Prediction data uploaded.")

# Run workflow
if st.button("Run Workflow"):
    st.write("🚀 Executing agents...")
    orchestrator = OrchestratorAgent(os.path.join(workflow_dir, selected_workflow))
    orchestrator.run()
    st.success("Workflow completed. Check terminal for output.")

st.markdown("---")
st.caption("Built with ♥ using Streamlit + Cursor-ready Agents")
