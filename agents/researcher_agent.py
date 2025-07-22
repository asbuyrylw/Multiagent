
from agents.base_agent import BaseAgent
from tools.search_tool import search_web

class ResearcherAgent(BaseAgent):
    def handle(self, task):
        query = task.get("query")
        results = search_web(query)
        return {"urls": results}
