import inspect
from enum import Enum
from pathlib import Path
from typing import Type, Any, Optional, List

import gymnasium as gym
from pydantic import BaseModel
from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.base_class import BaseAlgorithm

ALGORITHMS: List[Type[BaseAlgorithm]] = [PPO, SAC, TD3, DDPG, A2C, DQN]


def get_full_field_name(target_field: str, nested_model_type: Type[BaseModel], prefix: str = "") -> str:
    for field_name, field in nested_model_type.model_fields.items():
        current_path = f"{prefix}.{field_name}" if prefix else field_name

        # match leaf field
        if field_name == target_field:
            return current_path

        # recurse into nested BaseModel (assuming is it not optional or union)
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            result = get_full_field_name(target_field, field.annotation, current_path)
            if result:
                return result

    return ""


def is_enum_argument(arg_type: Any) -> bool:
    return inspect.isclass(arg_type) and issubclass(arg_type, Enum)


def get_drl_model(
        resume_from_path: Optional[Path],
        env: gym.Env,
        default_drl_model: BaseAlgorithm
) -> BaseAlgorithm:
    if resume_from_path is not None:
        if resume_from_path.exists():
            print(f"Loading already trained model from {resume_from_path}")
            return get_algorithm_class(resume_from_path).load(resume_from_path, env=env)
        else:
            raise ValueError(f"Resume path does not exist: {resume_from_path}")

    print("Loading default model")
    return default_drl_model


def get_algorithm_class(resume_from_path: Path) -> Type[BaseAlgorithm]:
    """
    This function returns the relevant algorithm name for resuming a pretrained model.
    Since storing the model is not leaving signs of the algorithm that was being used in the training process,
    the convention here is that the file's name (where the model to resume is saved)
    ** MUST include the name of the algorithm. **
    """
    for algo in ALGORITHMS:
        if algo.__name__.lower() in resume_from_path.stem.lower():
            return algo

    raise ValueError(f"The file name must include the name of one of the supported algorithms: {ALGORITHMS}")
