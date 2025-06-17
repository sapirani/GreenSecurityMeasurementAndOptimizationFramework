from dataclasses import dataclass

PRIME_MIN_VAL = 2 ** 1023 - 1
PRIME_MAX_VAL = 2 ** 1024 - 1

@dataclass
class KeyDetails:
    public_key: dict[str, int]
    private_key: dict[str, int]