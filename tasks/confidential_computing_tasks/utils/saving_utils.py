import pickle
from typing import Any

from tasks.confidential_computing_tasks.action_type import ActionType


def extract_messages_from_file(messages_file: str) -> list[int]:
    messages = []
    try:
        with open(messages_file, "r") as messages_file:
            messages = [int(msg.strip()) for msg in messages_file.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")

    if len(messages) == 0:
        raise Exception("No messages found. Must be at least one message.")

    return messages


def write_messages_to_file(messages_file: str, messages: list[int]) -> None:
    try:
        with open(messages_file, 'w') as f:
            for msg in messages:
                f.write(str(msg) + '\n')
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")
