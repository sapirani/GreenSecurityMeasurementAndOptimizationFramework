from typing import Optional

from typing_extensions import override

from tasks.confidential_computing_tasks.abstract_security_algorithm import T, SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.utils.checkpoint_storage.checkpoint_storage import CheckpointStorage
from tasks.confidential_computing_tasks.utils.saving_utils import save_checkpoint_file


class OperationCheckpointStorage(CheckpointStorage):
    def __init__(self, alg: SecurityAlgorithm, results_path: str, transformed_messages: list, action_type: ActionType,
                 initial_message_index: int):
        super().__init__(alg, results_path, transformed_messages, action_type, initial_message_index)
        self._total: Optional[T] = None

    def update(self, transformed_messages: list[T], total: T):
        self._total = total
        self.transformed_messages = transformed_messages

    @override
    def save_checkpoint(self):
        self._save_transformed_messages()
        serializable_total = self.alg.serialize_message(self._total)
        save_checkpoint_file(index=self.last_message_index + 1, total=serializable_total)

    @property
    def checkpoint_total(self) -> Optional[T]:
        return self._total
