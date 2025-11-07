import threading
from contextlib import asynccontextmanager
from typing import Annotated, List

import uvicorn
from dependency_injector.wiring import inject, Provide
from fastapi import FastAPI, Depends, Query
from consts import ElasticIndex
from hadoop_optimizer.container.container import Container
from hadoop_optimizer.drl_model.drl_model import DRLModel
from user_input.elastic_reader_input.abstract_date_picker import TimePickerChosenInput
from elastic_reader.main import main


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Elastic reader in the background")
    drl_model = app.container.drl_model()
    time_picker_input = app.container.drl_time_picker_input()
    indices_to_read_from = app.container.config.indices_to_read_from()
    t = threading.Thread(target=run_reader, args=(drl_model, time_picker_input, indices_to_read_from), daemon=True)
    t.start()
    yield
    # shutdown code
    print("Cleaning up tasks")

    # TODO: SUPPORT A threading.Event() TO SIGNAL STOP ITERATION
    t.join()


def run_reader(
        drl_model: DRLModel,
        time_picker_input: TimePickerChosenInput,
        indices_to_read_from: List[ElasticIndex]
):
    main(
        time_picker_input=time_picker_input,
        consumers=[drl_model],
        indices_to_read_from=indices_to_read_from
    )


app = FastAPI(lifespan=lifespan)


@app.get("/choose_configuration")
@inject
def choose_the_best_configuration_for_a_new_task_under_the_current_load(
    drl_model: Annotated[DRLModel, Depends(Provide[Container.drl_model])],
    param1: float = Query(...),
    param2: float = Query(...),
):
    return drl_model.get_best_configuration(param1=param1, param2=param2)


if __name__ == '__main__':
    container = Container()
    container.config.indices_to_read_from.from_value([ElasticIndex.PROCESS, ElasticIndex.SYSTEM])
    container.config.drl_state.split_by.from_value("hostname")
    container.config.drl_state.time_windows_seconds.from_value([1 * 60, 5 * 60, 10 * 60, 20 * 60])
    container.wire(modules=[__name__])
    app.container = container

    uvicorn.run(app, host="0.0.0.0", port=8000)
