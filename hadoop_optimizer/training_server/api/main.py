import subprocess
import time
from fastapi import Query, Body, HTTPException
from typing import Annotated, Optional
import uvicorn
from fastapi import FastAPI, Depends
from starlette import status

from DTOs.hadoop.hadoop_job import HadoopJob
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_types import JobType
from DTOs.hadoop.training_metadata import TrainingMetadata
from DTOs.hadoop.training_run_job_response import TrainingJobRunResponse
from hadoop_optimizer.nodes_trigger_sender import NodesTriggerSender
from hadoop_optimizer.supported_jobs.supported_jobs_config import SupportedJobsConfig
from hadoop_optimizer.training_server.api.config import MAX_JOB_RUNTIME


app = FastAPI()


def get_job_descriptor(
    job_type: JobType = Query(..., description="Select the type of Hadoop job to run."),
    input_size_gb: float = Query(..., gt=0, description="Input size in GB, must be >0")
) -> JobDescriptor:
    return JobDescriptor(job_type=job_type, input_size_gb=input_size_gb)


def get_scanner_extras(
    episode_num: Optional[int] = Query(
        None,
        ge=0,
        description="The index of the episode within the entire training process, must be >0"
    ),
    step_num: Optional[int] = Query(
        None,
        ge=0,
        description="The index of the current step withing the episode, must be >0"
    ),
    is_baseline: Optional[bool] = Query(
        None,
        description="Whether this run establishes the baseline performance for the current episode or not"
    )
) -> Optional[TrainingMetadata]:
    fields = [episode_num, step_num, is_baseline]

    if all(f is None for f in fields):
        return None
    if all(f is not None for f in fields):
        return TrainingMetadata(episode_num=episode_num, step_num=step_num, is_baseline=is_baseline)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=f"Either provide all of the following fields: {fields}, or none of them"
    )


def get_session_id(
    measurement_session_id: Optional[str] = Query(None, description="A custom provided session id for the scanner")
) -> Optional[str]:
    return measurement_session_id


@app.post("/run_job")
def run_selected_job_within_the_digital_twin_environment(
    job_descriptor: JobDescriptor = Depends(get_job_descriptor),
    session_id: Optional[str] = Depends(get_session_id),
    scanner_extras: Optional[TrainingMetadata] = Depends(get_scanner_extras),
    job_execution_config: HadoopJobExecutionConfig = Annotated[
        HadoopJobExecutionConfig,
        Body(
            ...,
            description="Select the desired Hadoop execution configuration.",
        )
    ]
) -> TrainingJobRunResponse:

    job_definition = SupportedJobsConfig.extract_job_definition(job_descriptor)
    selected_job = HadoopJob(
        job_definition=job_definition,
        job_execution_config=job_execution_config,
    )

    # TODO: BETTER NAMING?
    nodes_trigger_sender = NodesTriggerSender()
    scanner_logging_extras = scanner_extras.model_dump() if scanner_extras else {}

    try:
        nodes_trigger_sender.start_measurement(session_id=session_id, scanner_logging_extras=scanner_logging_extras)
        print("running job:", selected_job)
        start_time = time.perf_counter()
        subprocess.run(
            selected_job.get_hadoop_job_args(),
            check=True,
            timeout=MAX_JOB_RUNTIME,
        )
        runtime = time.perf_counter() - start_time
        nodes_trigger_sender.stop_measurement(session_id=session_id)
        return TrainingJobRunResponse(runtime_sec=runtime)
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,  # Gateway Timeout
            detail=f"Hadoop job exceeded {MAX_JOB_RUNTIME} seconds timeout"
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hadoop job failed: {e.stderr.strip() if e.stderr else 'Unknown error'}"
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Hadoop executable not found: {str(e)}"
        )


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
