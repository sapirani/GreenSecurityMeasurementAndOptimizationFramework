import threading
from contextlib import contextmanager
from pathlib import Path
from typing import List

from dependency_injector.wiring import inject, Provide
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback

from elastic_consumers.elastic_aggregations_logger import ElasticAggregationsLogger
from elastic_reader.consts import ElasticIndex
from elastic_reader_parameters import ES_URL, ES_PASS, ES_USER
from hadoop_optimizer.drl_telemetry.energy_tracker import EnergyTracker
from hadoop_optimizer.training_server.container.training_container import TrainingContainer
from elastic_reader.main import run_elastic_reader


@contextmanager
@inject
def run_energy_tracker(
        energy_tracker: EnergyTracker = Provide[TrainingContainer.energy_tracker],
        time_picker_input: EnergyTracker = Provide[TrainingContainer.drl_time_picker_input],
        elastic_aggregations_logger: ElasticAggregationsLogger = Provide[TrainingContainer.elastic_aggregations_logger],
        indices_to_read_from: List[ElasticIndex] = Provide[TrainingContainer.config.indices_to_read_from]
):
    print("Starting Elastic Reader")
    should_terminate_event = threading.Event()
    t = threading.Thread(
        target=run_elastic_reader,
        args=(time_picker_input, [energy_tracker, elastic_aggregations_logger], indices_to_read_from),
        kwargs=dict(should_terminate_event=should_terminate_event),
        daemon=True
    )
    t.start()
    yield
    # shutdown code
    print("terminating elastic reader")
    should_terminate_event.set()
    t.join()


@inject
def main(
        training_drl_model: BaseAlgorithm = Provide[TrainingContainer.training_drl_model],
        drl_model_storage_path: Path = Provide[TrainingContainer.config.drl_model_storage_path],
        learning_total_timestamps: int = Provide[TrainingContainer.config.learning_total_timestamps],
        save_freq: int = Provide[TrainingContainer.config.save_freq],
) -> None:
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,  # save every 'save_freq' steps
        save_path="./models/",
        name_prefix="ppo"
    )

    with run_energy_tracker():
        training_drl_model.learn(
            total_timesteps=learning_total_timestamps,
            log_interval=1,
            progress_bar=True,
            callback=checkpoint_callback
        )
        training_drl_model.save(drl_model_storage_path)


if __name__ == '__main__':
    container = TrainingContainer()
    container.config.drl_model_storage_path.from_value(Path("trained_ppo"))
    container.config.max_episode_steps.from_value(20)
    container.config.save_freq.from_value(10)
    container.config.learning_total_timestamps.from_value(100)
    container.config.alpha_hyperparam.from_value(1)
    container.config.beta_hyperparam.from_value(1)
    container.config.lambda_hyperparam.from_value(50)
    container.config.epsilon_hyperparam.from_value(2)
    container.config.tau_hyperparam.from_value(0.5)
    container.config.delta_hyperparam.from_value(0.5)
    container.config.elastic_username.from_value(ES_USER)
    container.config.elastic_password.from_value(ES_PASS)
    container.config.elastic_url.from_value(ES_URL)
    container.config.indices_to_read_from.from_value([ElasticIndex.PROCESS, ElasticIndex.SYSTEM])
    container.wire(modules=[__name__])
    main()
