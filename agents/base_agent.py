
class BaseAgent:
    def __init__(self, name, tools=None, memory=None):
        self.name = name
        self.tools = tools or []
        self.memory = memory

    def handle(self, task):
        raise NotImplementedError("Subclasses must implement this method.")
