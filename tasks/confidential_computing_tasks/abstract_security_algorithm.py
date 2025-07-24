import json
import math
import os
import pickle
import threading
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable, Generator

from sentry_sdk.serializer import serialize

from tasks.confidential_computing_tasks.key_details import KeyDetails, PRIME_MIN_VAL, PRIME_MAX_VAL

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
        serializable_messages = self._get_serializable_encrypted_messages(encrypted_messages)
        try:
            mode = "wb" if should_override_file else "ab"
            with open(file_name, mode) as messages_file:
                pickle.dump(serializable_messages, messages_file)
        except FileNotFoundError:
            print("Something went wrong with saving the encrypted messages")

    def load_encrypted_messages(self, file_name: str, starting_index: int) -> Generator[T, None, None]:
        try:
            file_idx = 1
            while True:
                file_parts = file_name.split(".")
                current_file = f"{file_parts[0]}{file_idx}{file_idx}.{file_parts[1]}"
                if os.path.exists(current_file):
                    with open(f"{current_file}", 'rb') as messages_file:
                        while True:
                            try:
                                encrypted_messages_portion = pickle.load(messages_file)
                                print("LEN: {}".format(len(encrypted_messages_portion)))
                                if len(encrypted_messages_portion) > starting_index:
                                    encrypted_messages_portion = encrypted_messages_portion[starting_index:]
                                    starting_index = 0
                                    for encrypted_msg in encrypted_messages_portion:
                                        deserialized_message = self.deserialize_message(encrypted_msg)
                                        yield deserialized_message
                                else:
                                    starting_index -= len(encrypted_messages_portion)
                            except EOFError:
                                break

                        file_idx += 1
                        if file_idx == 4:
                            break
                else:
                    break

        except Exception as e:
            print("Something went wrong with loading the encrypted messages", e)

    # def load_encrypted_messages(self, file_name: str) -> list[T]:
    #     try:
    #         with open(file_name, 'rb') as messages_file:
    #             deserialized_messages = []
    #             while True:
    #                 try:
    #                     encrypted_messages_portion = pickle.load(messages_file)
    #                     deserialized_messages_portion = self._get_deserializable_encrypted_messages(encrypted_messages_portion)
    #                     deserialized_messages.extend(deserialized_messages_portion)
    #                 except EOFError:
    #                     break
    #
    #             return deserialized_messages
    #     except FileNotFoundError:
    #         print("Something went wrong with loading the encrypted messages")
    #     raise RuntimeError("Could not load the encrypted messages.")

    def calc_encrypted_sum(self, messages: list[int], done_event: threading.Event, start_total: Optional[T] = None, checkpoint_callback: Optional[Callable[[int, T], None]] = None) -> T:
        regular_sum = sum(messages)
        return self.encrypt_message(regular_sum)

    def calc_encrypted_multiplication(self, messages: list[int], done_event: threading.Event, start_total: Optional[T] = None, checkpoint_callback: Optional[Callable[[int, T], None]] = None) -> T:
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