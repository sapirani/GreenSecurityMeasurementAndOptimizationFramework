import pickle
from typing import Any

from tasks.confidential_computing_tasks.action_type import ActionType

LAST_MESSAGE_INDEX_FILE_PATH = r"C:\Users\sapir\Desktop\last_message_index.txt"

def get_last_message_index() -> int:
    try:
        with open(LAST_MESSAGE_INDEX_FILE_PATH, "r") as last_message_file:
            content = last_message_file.read().strip()
            if content is None or content == "":
                return 0

            last_index = int(content)
            if last_index < 0 or last_index >= len(content):
                return 0
            
            return last_index
    except FileNotFoundError:
        return 0

def write_last_message_index(last_message_index: int):
    try:
        with open(LAST_MESSAGE_INDEX_FILE_PATH, "w") as last_message_file:
            last_message_file.write(str(last_message_index))
    except Exception as e:
        print(f"Failed to write last message index. The last index: {last_message_index}. The error: {e}")

def extract_messages_from_file(messages_file: str) -> list[int]:
    messages = []
    try:
        with open(messages_file, "r") as f:
            messages = [int(msg.strip()) for msg in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")

    if len(messages) == 0:
        raise Exception("No messages found. Must be at least one message.")

    last_message_index = get_last_message_index()
    return messages[last_message_index:]


def write_messages_to_file(messages_file: str, messages: list[int]) -> None:
    try:
        with open(messages_file, 'w') as f:
            for msg in messages:
                f.write(str(msg) + '\n')
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")
