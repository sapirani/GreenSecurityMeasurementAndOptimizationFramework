from datetime import datetime
import os
from pathlib import Path
from typing import List
from dependency_injector.wiring import inject, Provide
from human_id import generate_id
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from DTOs.logging.consts import IndexName
from elastic_reader.elastic_reader_parameters import ES_URL, ES_PASS, ES_USER
from hadoop_optimizer.training_loop.container.training_container import TrainingContainer, MODELS_DIR_NAME
from hadoop_optimizer.optimization_mode import OptimizationMode

MINUTE = 60


@inject
def main(
        drl_training_model: BaseAlgorithm = Provide[TrainingContainer.drl_training_model],
        learning_total_timestamps: int = Provide[TrainingContainer.config.drl.learning_total_timestamps],
        training_callback: BaseCallback = Provide[TrainingContainer.training_callback],
        final_model_saving_paths: List[str] = Provide[TrainingContainer.final_model_saving_paths],
) -> None:
    """
    This function runs a drl training and store the final + intermediate models.
    Depending on TrainingContainer.config.drl.resume_from_path, this function eiter train a fresh model, or resuming
    a previously trained model and keeps on training it.
    Note: the file names that store the models, must include the model name (e.g., PPO, A2C, etc.).
    The reason for this convention is that the automatic code that resumes the pretrained model must know its type
    """
    drl_training_model.learn(
        total_timesteps=learning_total_timestamps,
        log_interval=1,
        progress_bar=True,
        callback=training_callback
        # TODO: CONSIDER SETTING reset_num_timesteps TO FALSE TO START THE INTERNAL COUNTING FROM THE SAME POINT
        #  WHERE THE PRETRAINED MODEL STOPPED
    )

    for saving_path in final_model_saving_paths:
        drl_training_model.save(saving_path)


if __name__ == '__main__':
    container = TrainingContainer()
    # Use a path to a pretrained model, or None if you want to start training all over again
    container.config.drl.resume_from_path.from_value(
        Path(os.path.dirname(os.path.abspath(__file__))) /
        Path(MODELS_DIR_NAME) /
        Path("trained_PPO.zip")
    )
    container.config.drl.train_id.from_value(generate_id(word_count=3))
    container.config.drl.mode.from_value(OptimizationMode.CONTEXTUAL_BANDIT)
    container.config.drl.storage.models_base_dir.from_value(os.path.dirname(os.path.abspath(__file__)))
    container.config.drl.storage.save_freq.from_value(2048)
    container.config.drl.env.max_episode_steps.from_value(50)
    container.config.drl.env.truncated_penalty.from_value(-150.0)
    container.config.drl.state.split_by.from_value("hostname")
    container.config.drl.state.leverage_telemetry_in_state.from_value(False)
    container.config.drl.state.time_windows_seconds.from_value([1 * MINUTE, 5 * MINUTE, 10 * MINUTE, 20 * MINUTE])
    container.config.drl.cached_results.search_since.from_value(datetime(year=2026, month=5, day=29))
    container.config.drl.cached_results.force_real_execution_probability.from_value(0.001)
    container.config.drl.cached_results.utilization_policy.max_param_diff_percent.from_value(27)
    container.config.drl.cached_results.utilization_policy.min_required_similar_samples.from_value(3)
    container.config.drl.cached_results.utilization_policy.results_noise_scale.from_value(0.3)
    container.config.drl.cached_results.utilization_policy.similarity_temperature.from_value(0.12)
    container.config.drl.cached_results.utilization_policy.running_time_max_deviation_percent.from_value(12)
    container.config.drl.cached_results.utilization_policy.energy_max_deviation_percent.from_value(18)
    container.config.drl.learning_total_timestamps.from_value(120000)
    container.config.drl.reward.alpha.from_value(1)
    container.config.drl.reward.beta.from_value(1)
    container.config.drl.reward.lambda_.from_value(50)
    container.config.drl.reward.epsilon.from_value(2)
    container.config.drl.reward.tau.from_value(0.05)
    container.config.drl.reward.delta.from_value(0.95)
    container.config.drl.algorithm.hyperparameters.n_steps.from_value(512)
    container.config.drl.algorithm.hyperparameters.batch_size.from_value(64)
    container.config.drl.algorithm.hyperparameters.n_epochs.from_value(5)
    container.config.drl.algorithm.hyperparameters.gamma.from_value(1)  # automatically converted to 0 in contextual bandit mode
    container.config.drl.algorithm.hyperparameters.ent_coef.from_value(0.1)   # encourage exploration
    container.config.drl.algorithm.hyperparameters.use_sde.from_value(True)
    # Resample the gSDE noise matrix every <sde_sample_freq> steps.
    # -1 would keep the same noise matrix for the entire rollout.
    container.config.drl.algorithm.hyperparameters.sde_sample_freq.from_value(1)
    container.config.drl.policy.hyperparameters.net_arch.from_value([128, 128])
    container.config.drl.policy.hyperparameters.squash_output.from_value(True)
    container.config.drl.policy.hyperparameters.log_std_init.from_value(-0.5)
    container.config.elastic.username.from_value(ES_USER)
    container.config.elastic.password.from_value(ES_PASS)
    container.config.elastic.url.from_value(ES_URL)
    container.config.elastic.indices_to_read_from.from_value([IndexName.PROCESS_METRICS, IndexName.SYSTEM_METRICS])
    container.wire(modules=[__name__])
    main()
