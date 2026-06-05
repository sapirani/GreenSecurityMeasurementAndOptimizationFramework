import inspect
from enum import Enum
from typing import Type, Any

from pydantic import BaseModel


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