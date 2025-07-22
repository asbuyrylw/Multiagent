
import unittest
from agents.orchestrator_agent import OrchestratorAgent

class TestWorkflowRun(unittest.TestCase):
    def test_end_to_end_workflow(self):
        workflow_path = "workflows/solar_leads_workflow.yaml"
        orchestrator = OrchestratorAgent(workflow_path)
        orchestrator.run()
        self.assertTrue(True)  # If no exception, test passes

if __name__ == "__main__":
    unittest.main()
