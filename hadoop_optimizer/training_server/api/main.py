import subprocess
import time
from fastapi import Query, Body, HTTPException
from typing import Annotated
import uvicorn
from fastapi import FastAPI, Depends
from starlette import status

from DTOs.hadoop.hadoop_job import HadoopJob
from DTOs.hadoop.hadoop_job_definition import HadoopJobDefinition
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_types import JobType
from DTOs.hadoop.training_run_job_response import TrainingJobRunResponse
from hadoop_optimizer.supported_jobs.supported_jobs_config import SupportedJobsConfig
from hadoop_optimizer.training_server.api.config import MAX_JOB_RUNTIME

app = FastAPI()


def get_job_descriptor(
    job_type: JobType = Query(..., description="Select the type of Hadoop job to run."),
    input_size_gb: float = Query(..., gt=0, description="Input size in GB, must be >0")
) -> JobDescriptor:
    return JobDescriptor(job_type=job_type, input_size_gb=input_size_gb)


@app.post("/run_job")
def run_selected_job_within_the_digital_twin_environment(
    job_descriptor: JobDescriptor = Depends(get_job_descriptor),
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

    try:
        print("running job:", selected_job)
        start_time = time.perf_counter()
        subprocess.run(
            # selected_job.get_hadoop_job_args(),
            ["python", "-c", "import time; time.sleep(30)"],
            check=True,
            timeout=MAX_JOB_RUNTIME,
        )
        runtime = time.perf_counter() - start_time
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
