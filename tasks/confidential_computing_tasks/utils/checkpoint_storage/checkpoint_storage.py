import warnings

from tasks.confidential_computing_tasks.abstract_seurity_algorithm import SecurityAlgorithm
from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.utils.algorithm_utils import is_new_execution
from tasks.confidential_computing_tasks.utils.saving_utils import write_last_message_index, write_messages_to_file


class CheckpointStorage:
    def __init__(self, alg: SecurityAlgorithm, results_path: str, transformed_messages: list, action_type: ActionType,
                 initial_message_index: int):
        self.alg = alg
        self.results_path = results_path
        self.transformed_messages = transformed_messages
        self.action_type = action_type
        self.should_override = is_new_execution(initial_message_index)
        self.initial_message_index = initial_message_index

    @property
    def last_message_index(self):
        return len(self.transformed_messages) + self.initial_message_index

    def save_transformed_messages(self):
        if self.last_message_index is not None and self.last_message_index > 0:
            write_last_message_index(self.last_message_index)
        if self.transformed_messages is not None and len(self.transformed_messages) > 0 and self.results_path != "":
            print("SHOULD OVERRIDE SAVING PATH (in class):", self.should_override)
            if self.action_type == ActionType.Encryption:
                self.alg.save_encrypted_messages(self.transformed_messages, self.results_path, should_override_file=self.should_override)
            # If decryption or full pipeline, optionally save decrypted ints as text
            elif self.action_type in (ActionType.Decryption, ActionType.FullPipeline):
                write_messages_to_file(self.results_path, self.transformed_messages, should_override_file=self.should_override)
            else:
                warnings.warn("The transformed messages weren't saved in a file.", RuntimeWarning)

