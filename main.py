
from agents.orchestrator_agent import OrchestratorAgent
from utils.logger import setup_logger

if __name__ == "__main__":
    logger = setup_logger("main")
    workflow_file = "workflows/solar_leads_workflow.yaml"
    logger.info(f"Executing workflow: {workflow_file}")
    orchestrator = OrchestratorAgent(workflow_file)
    orchestrator.run()
