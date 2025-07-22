
from pydantic import BaseModel

class TaskInput(BaseModel):
    query: str
