from dataclasses import dataclass


@dataclass
class KeyDetails:
    public_key: dict[str, int]
    private_key: dict[str, int]