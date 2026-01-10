from pydantic import BaseModel


class TrainingJobRunResponse(BaseModel):    # todo: is it supposed to be written here?
    runtime_sec: float
