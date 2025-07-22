
import yaml
from agents.researcher_agent import ResearcherAgent
from agents.scraper_agent import ScraperAgent
from agents.ml_agent import MLAgent
from utils.logger import setup_logger

class OrchestratorAgent:
    def __init__(self, workflow_path):
        self.logger = setup_logger("orchestrator")
        self.workflow = self.load_workflow(workflow_path)
        self.agents = {
            "researcher": ResearcherAgent("researcher"),
            "scraper": ScraperAgent("scraper"),
            "ml": MLAgent("ml")
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
            input_data = step.get("input", state.get(source, {}))
            self.logger.info(f"Running {target} with input: {input_data}")
            result = agent.handle(input_data)
            self.logger.info(f"{target} result: {result}")
            state[target] = result
