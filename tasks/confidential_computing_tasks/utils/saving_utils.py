import os
import pickle
from typing import Any

from tasks.confidential_computing_tasks.action_type import ActionType

LAST_MESSAGE_INDEX_FILE_PATH = r"C:\Users\Administrator\Desktop\last_message_index.txt"


def get_last_message_index() -> int:
    try:
        with open(LAST_MESSAGE_INDEX_FILE_PATH, "r") as last_message_file:
            content = last_message_file.read().strip()
            if content is None or content == "":
                return 0

            last_index = int(content)
            if last_index < 0:
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

    return messages


def write_messages_to_file(messages_file: str, messages: list[int], should_override_file: bool) -> None:
    try:
        mode = "w" if should_override_file else "a"
        with open(messages_file, mode) as f:
            for msg in messages:
                f.write(str(msg) + '\n')
    except FileNotFoundError:
        raise FileNotFoundError("Messages file not found.")


def save_checkpoint_file(index: int, total):
    with open("checkpoint.pkl", "wb") as f:
        pickle.dump({"index": index, "total": total}, f)


def read_checkpoint_file() -> tuple:
    if not os.path.exists("checkpoint.pkl"):
        return 0, None
    with open("checkpoint.pkl", "rb") as f:
        checkpoint = pickle.load(f)
        return checkpoint["index"], checkpoint["total"]
