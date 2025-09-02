from dataclasses import dataclass


@dataclass(frozen=True)
class SessionHostIdentity:
    hostname: str
    session_id: str
