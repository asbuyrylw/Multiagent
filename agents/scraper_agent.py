
from agents.base_agent import BaseAgent
from tools.scrape_tool import scrape_url

class ScraperAgent(BaseAgent):
    def handle(self, task):
        urls = task.get("urls", [])
        texts = []
        for url in urls:
            text = scrape_url(url)
            texts.append(text)
        return {"text_data": texts}
