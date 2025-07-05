from typing import Optional

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import T
from tasks.confidential_computing_tasks.utils.checkpoint_storage.checkpoint_storage import CheckpointStorage


class OperationCheckpointStorage(CheckpointStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index: int = self.initial_message_index
        self._total: Optional[T] = None

    def update(self, index: int, total: T):
        self._index = index
        self._total = total

    @property
    def checkpoint_index(self) -> int:
        return self._index

    @property
    def checkpoint_total(self) -> Optional[T]:
        return self._total