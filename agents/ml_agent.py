
from agents.base_agent import BaseAgent
from tools.ml_tool import train_and_predict

class MLAgent(BaseAgent):
    def handle(self, task):
        return train_and_predict(task)
