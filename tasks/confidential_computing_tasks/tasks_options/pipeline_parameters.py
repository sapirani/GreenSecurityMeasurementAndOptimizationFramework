from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineParameters:
    """docstring for PipelineParameters."""
    path_for_messages: str
    path_for_result_messages: str
    key_file: str
    min_key_value: int
    max_key_value: int
    encryption_algorithm: int
    cipher_block_mode: Optional[int]