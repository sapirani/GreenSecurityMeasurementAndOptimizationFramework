from pydantic import BaseModel


class TrainingJobRunResponse(BaseModel):
    runtime_sec: float
