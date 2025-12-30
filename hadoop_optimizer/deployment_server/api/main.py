import threading
from contextlib import asynccontextmanager
from fastapi import HTTPException
from typing import Annotated, List, Optional
import uvicorn
from dependency_injector.wiring import inject, Provide
from fastapi import FastAPI, Depends, Request
from stable_baselines3.common.base_class import BaseAlgorithm
from starlette import status
from starlette.responses import JSONResponse
from elastic_reader.consts import ElasticIndex
from hadoop_optimizer.DTOs.hadoop_job_config import HadoopJobConfig
from hadoop_optimizer.DTOs.job_properties import JobProperties, get_job_properties
from hadoop_optimizer.deployment_server.container.container import Container
from hadoop_optimizer.deployment_server.drl_model.drl_model import DRLModel
from hadoop_optimizer.deployment_server.drl_model.drl_state import StateNotReadyException
from hadoop_optimizer.drl_envs.deployment_env import OptimizerDeploymentEnv
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput
from elastic_reader.main import run_elastic_reader


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Elastic reader in the background")
    drl_model = app.container.drl_model()
    time_picker_input = app.container.drl_time_picker_input()
    indices_to_read_from = app.container.config.indices_to_read_from()
    should_terminate_event = threading.Event()
    t = threading.Thread(
        target=run_telemetry_reader,
        args=(drl_model, time_picker_input, indices_to_read_from, should_terminate_event),
        daemon=True
    )
    t.start()
    yield
    # shutdown code
    print("Cleaning up tasks")
    should_terminate_event.set()
    t.join()


def run_telemetry_reader(
        drl_model: DRLModel,
        time_picker_input: TimePickerChosenInput,
        indices_to_read_from: List[ElasticIndex],
        should_terminate_event: Optional[threading.Event] = None
):
    run_elastic_reader(
        time_picker_input=time_picker_input,
        consumers=[drl_model],
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

# TODO: INCORPORATE PREVIOUS DRL STATE CODE FOR CLUSTER LOAD
# @app.get("/choose_configuration")
# @inject
# def choose_the_best_configuration_for_a_new_task_under_the_current_load(
#     job_properties: Annotated[JobProperties, Depends(get_job_properties)],
#     drl_model: Annotated[DRLModel, Depends(Provide[Container.drl_model])],
# ):
#     return drl_model.determine_best_job_configuration(job_properties)


class EnvironmentTruncatedException(Exception):
    def __init__(self, last_job_configuration: HadoopJobConfig, elapsed_steps: int, max_steps: int):
        self.last_job_configuration = last_job_configuration
        self.elapsed_steps = elapsed_steps
        self.max_steps = max_steps
        super().__init__("Environment truncated or step called incorrectly")


def determine_best_job_configuration(
        deployment_agent: BaseAlgorithm,
        deployment_env: OptimizerDeploymentEnv,
        job_properties: JobProperties
) -> HadoopJobConfig:
    with deployment_env:
        print("understanding shape:")
        print(deployment_env.observation_space)
        print(deployment_env.observation_space.shape)
        obs, _ = deployment_env.reset(options=job_properties.model_dump())
        while True:
            action, _states = deployment_agent.predict(obs)
            obs, rewards, terminated, truncated, info = deployment_env.step(action)
            deployment_env.render()

            if terminated:
                return HadoopJobConfig.model_validate(info["current_hadoop_config"])   # TODO: USE A CONST FOR KEY NAME?

            if truncated:
                raise EnvironmentTruncatedException(
                    HadoopJobConfig.model_validate(info["current_hadoop_config"]),
                    info["elapsed_steps"],
                    info["max_steps"],
                )


@app.get("/choose_configuration")
@inject
def choose_the_best_configuration_for_a_new_task_under_the_current_load(
    job_properties: Annotated[JobProperties, Depends(get_job_properties)],
    deployment_agent: Annotated[BaseAlgorithm, Depends(Provide[Container.deployment_agent])],
    deployment_env: Annotated[OptimizerDeploymentEnv, Depends(Provide[Container.deployment_env])],
) -> HadoopJobConfig:

    try:
        return determine_best_job_configuration(deployment_agent, deployment_env, job_properties)
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
    container = Container()
    container.config.max_episode_steps.from_value(100)
    container.config.indices_to_read_from.from_value([ElasticIndex.PROCESS, ElasticIndex.SYSTEM])
    container.config.drl_state.split_by.from_value("hostname")
    container.config.drl_state.time_windows_seconds.from_value([1 * 60, 5 * 60, 10 * 60, 20 * 60])
    container.wire(modules=[__name__])
    app.container = container

    uvicorn.run(app, host="0.0.0.0", port=8000)
