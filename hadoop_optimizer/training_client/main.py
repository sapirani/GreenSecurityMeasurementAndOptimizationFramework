from pathlib import Path
from dependency_injector.wiring import inject, Provide
from stable_baselines3.common.base_class import BaseAlgorithm
from hadoop_optimizer.training_server.container.training_container import TrainingContainer


@inject
def main(
        training_drl_model: BaseAlgorithm = Provide[TrainingContainer.training_drl_model],
        drl_model_storage_path: Path = Provide[TrainingContainer.config.drl_model_storage_path]
) -> None:
    training_drl_model.learn(total_timesteps=10, log_interval=1, progress_bar=True)
    training_drl_model.save(drl_model_storage_path)


if __name__ == '__main__':
    container = TrainingContainer()
    container.config.drl_model_storage_path.from_value(Path("trained_ppo"))
    container.config.max_episode_steps.from_value(5)
    # container.config.indices_to_read_from.from_value([ElasticIndex.PROCESS, ElasticIndex.SYSTEM])
    # container.config.drl_state.split_by.from_value("hostname")
    # container.config.drl_state.time_windows_seconds.from_value([1 * 60, 5 * 60, 10 * 60, 20 * 60])
    container.wire(modules=[__name__])
    main()
