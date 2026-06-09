import math
import os
import pickle
import threading
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable, Generator

from tasks.programs.confidential_computing_tasks.key_details import KeyDetails, PRIME_MIN_VAL, PRIME_MAX_VAL

T = TypeVar('T')


class SecurityAlgorithm(ABC, Generic[T]):
    def __init__(self, min_key_val: int = PRIME_MIN_VAL, max_key_val: int = PRIME_MAX_VAL):
        self._min_key_val = min_key_val
        self._max_key_val = max_key_val

    def serialize_message(self, msg: T) -> bytes:
        return pickle.dumps(msg)

    def deserialize_message(self, msg: bytes) -> T:
        return pickle.loads(msg)

    def save_encrypted_messages(self, encrypted_messages: list[T], file_name: str, should_override_file: bool):
        """
        Method that saves encrypted messages to a file. We first serialize the message and then write them to a file in bytes.
        Input:
            - encrypted_messages: a list of encrypted messages
            - file_name: the name of the file to save the encrypted messages to
            - should_override_file: whether or not to save the encrypted messages to a new file. If not, append the new messages to the end of the file.
        """
        serializable_messages = self._get_serializable_encrypted_messages(encrypted_messages)
        try:
            mode = "wb" if should_override_file else "ab"
            with open(file_name, mode) as messages_file:
                pickle.dump(serializable_messages, messages_file)
        except FileNotFoundError:
            print("Something went wrong with saving the encrypted messages")

    def load_encrypted_messages(self, file_name: str, starting_index: int) -> Generator[T, None, None]:
        """
        This method reads encrypted messages from file (that are serialized), and deserializes them.
        In order to handle large files, we read a portion of the messages (using pickle.load) and deserialize each message separately.
        The method can use a starting_index which mentions what was the index of the last message that was loaded (for cases where the computer shuts down but the experiment is not over).

        The partial reading is relevant in cases of calling `save_encrypted_messages` at least two times. Each time, list of encrypted messages is saved as a full object, so the portion matches such one list.
        Input:
            - file_name: the name of the file to load (the file that contains encrypted messages)
            - starting_index: the index of the message that should be loaded now
        Output:
            - Generator that yields deserialized messages
        """
        try:
            if os.path.exists(file_name):
                with open(f"{file_name}", 'rb') as messages_file:
                    while True:
                        try:
                            # the pickle.load method knows to read an entire object from the current place (pointer) in the file.
                            # the object that should be loaded is a list of messages
                            # this mechanism should handle cases of large files and avoid crashing over using too much memory
                            # for example, when the experiment of encrypting messages crushed at least once and there are a lot of messages to encrypt.
                            encrypted_messages_portion = pickle.load(messages_file)
                            print("LEN: {}".format(len(encrypted_messages_portion)))

                            # check if these messages were already read and deserialized in a previous experiment
                            if len(encrypted_messages_portion) > starting_index:
                                encrypted_messages_portion = encrypted_messages_portion[starting_index:]
                                starting_index = 0
                                for encrypted_msg in encrypted_messages_portion:
                                    deserialized_message = self.deserialize_message(encrypted_msg)
                                    yield deserialized_message
                            else:
                                starting_index -= len(encrypted_messages_portion)
                        except EOFError:
                            # when the pickle.load is done reading all objects in the file
                            break


            else:
                raise RuntimeError("No message found")

        except Exception as e:
            print("Something went wrong with loading the encrypted messages", e)

    def calc_encrypted_sum(self, messages: list[int], done_event: threading.Event, start_total: Optional[T] = None,
                           checkpoint_callback: Optional[Callable[[int, T], None]] = None) -> T:
        regular_sum = sum(messages)
        return self.encrypt_message(regular_sum)

    def calc_encrypted_multiplication(self, messages: list[int], done_event: threading.Event,
                                      start_total: Optional[T] = None,
                                      checkpoint_callback: Optional[Callable[[int, T], None]] = None) -> T:
        total_mul = math.prod(messages)
        return self.encrypt_message(total_mul)

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[T]) -> list[bytes]:
        return [self.serialize_message(msg) for msg in encrypted_messages]

    @abstractmethod
    def _generate_and_save_key(self, key_file) -> KeyDetails:
        pass

    @abstractmethod
    def _load_key(self, key_file) -> KeyDetails:
        pass

    def extract_key(self, key_file: str, should_generate: bool) -> KeyDetails:
        """ Initialize the public and private key """
        try:
            if should_generate:
                return self._generate_and_save_key(key_file)
            return self._load_key(key_file)
        except Exception as e:
            raise Exception("Something went wrong with extracting the key.")

    @abstractmethod
    def encrypt_message(self, msg: int) -> T:
        """ Encrypt the message """
        pass

    @abstractmethod
    def decrypt_message(self, msg: T) -> int:
        """ Decrypt the message """
        pass
