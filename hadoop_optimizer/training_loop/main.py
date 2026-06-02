import os
from pathlib import Path
from dependency_injector.wiring import inject, Provide
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from DTOs.logging.consts import IndexName
from elastic_reader.elastic_reader_parameters import ES_URL, ES_PASS, ES_USER
from hadoop_optimizer.training_loop.container.training_container import TrainingContainer


@inject
def main(
        training_drl_model: BaseAlgorithm = Provide[TrainingContainer.training_drl_model],
        drl_model_storage_path: Path = Provide[TrainingContainer.config.drl.storage.model_path],
        learning_total_timestamps: int = Provide[TrainingContainer.config.drl.learning_total_timestamps],
        save_freq: int = Provide[TrainingContainer.config.drl.storage.save_freq],
) -> None:
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        name_prefix="ppo"
    )

    training_drl_model.learn(
        total_timesteps=learning_total_timestamps,
        log_interval=1,
        progress_bar=True,
        callback=checkpoint_callback
    )
    training_drl_model.save(drl_model_storage_path)


if __name__ == '__main__':
    container = TrainingContainer()
    container.config.drl.storage.model_path.from_value(Path("trained_ppo"))
    container.config.drl.storage.save_freq.from_value(100)
    container.config.drl.env.max_episode_steps.from_value(50)
    container.config.drl.env.max_param_diff_percent.from_value(27)
    container.config.drl.learning_total_timestamps.from_value(2000)
    container.config.drl.reward.alpha.from_value(1)
    container.config.drl.reward.beta.from_value(1)
    container.config.drl.reward.lambda_.from_value(50)
    container.config.drl.reward.epsilon.from_value(2)
    container.config.drl.reward.tau.from_value(0.05)
    container.config.drl.reward.delta.from_value(0.95)
    container.config.elastic.username.from_value(ES_USER)
    container.config.elastic.password.from_value(ES_PASS)
    container.config.elastic.url.from_value(ES_URL)
    container.config.elastic.indices_to_read_from.from_value([IndexName.PROCESS_METRICS, IndexName.SYSTEM_METRICS])
    container.wire(modules=[__name__])
    main()
