from enum import Enum


class ActionType(Enum):
    Encryption = 1
    Decryption = 2
    FullPipeline = 3