from dataclasses import dataclass

PRIME_MIN_VAL = 2 ** 2047 - 1
PRIME_MAX_VAL = 2 ** 2048 - 1

@dataclass
class KeyDetails:
    public_key: dict[str, int]
    private_key: dict[str, int]