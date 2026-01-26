import threading
from contextlib import asynccontextmanager
from fastapi import HTTPException
from typing import Annotated, List, Optional
import uvicorn
from dependency_injector.wiring import inject, Provide
from fastapi import FastAPI, Depends, Request
from starlette import status
from starlette.responses import JSONResponse
from elastic_reader.consts import ElasticIndex
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_properties import JobProperties, get_job_properties
from hadoop_optimizer.deployment_server.container.deployment_container import DeploymentContainer
from hadoop_optimizer.deployment_server.drl_deployment_manager import DRLDeploymentManager
from hadoop_optimizer.drl_telemetry.telemetry_aggregator import TelemetryAggregator
from hadoop_optimizer.erros import EnvironmentTruncatedException, StateNotReadyException
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput
from elastic_reader.main import run_elastic_reader


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Elastic reader in the background")
    telemetry_aggregator = app.container.telemetry_aggregator()
    time_picker_input = app.container.drl_time_picker_input()
    indices_to_read_from = app.container.config.elastic.indices_to_read_from()
    should_terminate_event = threading.Event()
    t = threading.Thread(
        target=run_telemetry_reader,
        args=(telemetry_aggregator, time_picker_input, indices_to_read_from, should_terminate_event),
        daemon=True
    )
    t.start()
    yield
    # shutdown code
    print("Cleaning up tasks")
    should_terminate_event.set()
    t.join()


def run_telemetry_reader(
        telemetry_aggregator: TelemetryAggregator,
        time_picker_input: TimePickerChosenInput,
        indices_to_read_from: List[ElasticIndex],
        should_terminate_event: Optional[threading.Event] = None
):
    run_elastic_reader(
        time_picker_input=time_picker_input,
        consumers=[telemetry_aggregator],
        indices_to_read_from=indices_to_read_from,
        should_terminate_event=should_terminate_event
    )


app = FastAPI(lifespan=lifespan)


@app.exception_handler(StateNotReadyException)
async def state_not_ready_exception_handler(request: Request, exc: StateNotReadyException):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,  # Service Unavailable
        content={"detail": str(exc)}
    )


@app.get("/choose_configuration")
@inject
def choose_the_best_configuration_for_a_new_task_under_the_current_load(
    job_properties: Annotated[JobProperties, Depends(get_job_properties)],
    drl_deployment_manager: Annotated[DRLDeploymentManager, Depends(Provide[DeploymentContainer.drl_deployment_manager])],
) -> HadoopJobExecutionConfig:

    try:
        return drl_deployment_manager.determine_best_job_configuration(job_properties)
    except EnvironmentTruncatedException as e:
        # Return HTTP 400 (Bad Request) or 500 depending on semantics
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "elapsed_steps": e.elapsed_steps,
                "max_steps": e.max_steps,
                "last_job_configuration": e.last_job_configuration.model_dump(),
            }
        )


if __name__ == '__main__':
    container = DeploymentContainer()
    container.config.elastic.indices_to_read_from.from_value([ElasticIndex.PROCESS, ElasticIndex.SYSTEM])
    container.config.drl.env.max_episode_steps.from_value(100)
    container.config.drl.state.split_by.from_value("hostname")
    container.config.drl.state.time_windows_seconds.from_value([1 * 60, 5 * 60, 10 * 60, 20 * 60])
    container.wire(modules=[__name__])
    app.container = container

    uvicorn.run(app, host="0.0.0.0", port=8000)
