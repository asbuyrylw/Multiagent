
import yaml
from agents.researcher_agent import ResearcherAgent
from agents.scraper_agent import ScraperAgent
from agents.ml_agent import MLAgent
from agents.housing_data_agent import HousingDataAgent
from agents.housing_ml_agent import HousingMLAgent
from utils.logger import setup_logger

class OrchestratorAgent:
    def __init__(self, workflow_path):
        self.logger = setup_logger("orchestrator")
        self.workflow = self.load_workflow(workflow_path)
        self.agents = {
            "researcher": ResearcherAgent("researcher"),
            "scraper": ScraperAgent("scraper"),
            "ml": MLAgent("ml"),
            "housing_data": HousingDataAgent("housing_data"),
            "housing_ml": HousingMLAgent("housing_ml")
        }

    def load_workflow(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def run(self):
        state = {}
        for step in self.workflow["workflow"]:
            source = step["source"]
            target = step["target"]
            agent = self.agents[target]
            
            # Prepare input data
            if source == "input":
                input_data = step.get("input", {})
            else:
                # Pass data from previous agent
                source_data = state.get(source, {})
                step_input = step.get("input", {})
                input_data = {**source_data, **step_input}
            
            self.logger.info(f"Running {target} agent...")
            result = agent.handle(input_data)
            
            if isinstance(result, dict) and result.get('status') == 'error':
                self.logger.error(f"{target} failed: {result.get('message', 'Unknown error')}")
                return state
            
            self.logger.info(f"{target} completed successfully")
            state[target] = result
        
        return state
