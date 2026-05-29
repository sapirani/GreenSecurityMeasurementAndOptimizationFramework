from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from dependency_injector.wiring import inject, Provide
from fastapi import FastAPI, Depends, Request
from fastapi import HTTPException
from starlette import status
from starlette.responses import JSONResponse

from DTOs.hadoop.drl.job_properties import JobProperties, get_job_properties
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.logging.consts import IndexName
from hadoop_optimizer.common.erros import EnvironmentTruncatedException, StateNotReadyException
from hadoop_optimizer.job_config_recommender.server.container.deployment_container import DeploymentContainer
from hadoop_optimizer.job_config_recommender.server.drl_deployment_manager import DRLDeploymentManager

MINUTE = 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Elastic reader in the background")
    elastic_reader_service = app.container.elastic_reader_service()
    elastic_reader_service.start_in_background()
    try:
        yield
    finally:
        print("Stopping Elastic reader service")
        elastic_reader_service.stop()


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
    container.config.elastic.indices_to_read_from.from_value([IndexName.PROCESS_METRICS, IndexName.SYSTEM_METRICS])
    container.config.drl.env.max_episode_steps.from_value(100)
    container.config.drl.state.split_by.from_value("hostname")
    container.config.drl.state.time_windows_seconds.from_value([1 * MINUTE, 5 * MINUTE, 10 * MINUTE, 20 * MINUTE])
    container.wire(modules=[__name__])
    app.container = container

    uvicorn.run(app, host="0.0.0.0", port=8000)
