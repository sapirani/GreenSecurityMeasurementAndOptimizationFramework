import json
import math
import pickle
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable

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

    def load_encrypted_messages(self, file_name: str) -> list[T]:
        try:
            with open(file_name, 'rb') as messages_file:
                deserialized_messages = []
                while True:
                    try:
                        encrypted_messages_portion = pickle.load(messages_file)
                        deserialized_messages_portion = self._get_deserializable_encrypted_messages(encrypted_messages_portion)
                        deserialized_messages.extend(deserialized_messages_portion)
                    except EOFError:
                        break

                return deserialized_messages
        except FileNotFoundError:
            print("Something went wrong with loading the encrypted messages")
        raise RuntimeError("Could not load the encrypted messages.")

    def calc_encrypted_sum(self, messages: list[int], start_total: Optional[T] = None, checkpoint_callback: Optional[Callable[[int, T], None]] = None) -> T:
        regular_sum = sum(messages)
        return self.encrypt_message(regular_sum)

    def calc_encrypted_multiplication(self, messages: list[int], start_total: Optional[T] = None, checkpoint_callback: Optional[Callable[[int, T], None]] = None) -> T:
        total_mul = math.prod(messages)
        return self.encrypt_message(total_mul)

    def _get_serializable_encrypted_messages(self, encrypted_messages: list[T]) -> list[bytes]:
        return [self.serialize_message(msg) for msg in encrypted_messages]

    def _get_deserializable_encrypted_messages(self, serialized_encrypted_messages: list[bytes]) -> list[T]:
        return [self.deserialize_message(msg) for msg in serialized_encrypted_messages]

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