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

def save_encrypted_messages(encrypted_messages: list[Any], file_name: str):
    try:
        with open(file_name, 'rb') as messages_file:
            pickle.dump(encrypted_messages, messages_file)
    except FileNotFoundError:
        print("Something went wrong with saving the encrypted messages")

def load_encrypted_messages(file_name: str) ->  list:
    try:
        with open(file_name, 'rb') as messages_file:
            encrypted_messages = pickle.load(messages_file)
            return encrypted_messages
    except FileNotFoundError:
        print("Something went wrong with loading the encrypted messages")
        return []

def save_messages_after_action(updated_messages: list[Any], result_messages_file: str, action_type: ActionType):
    # If encryption, save encrypted messages to result file
    if action_type == ActionType.Encryption:
        save_encrypted_messages(updated_messages, result_messages_file)
    # If decryption or full pipeline, optionally save decrypted ints as text
    elif action_type in (ActionType.Decryption, ActionType.FullPipeline):
        write_messages_to_file(result_messages_file, updated_messages)
    else:
        raise Exception("Unknown action type.")