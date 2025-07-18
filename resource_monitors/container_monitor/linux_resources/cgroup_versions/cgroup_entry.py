from dataclasses import dataclass


@dataclass
class CgroupEntry:
    hierarchy_id: int
    subsystems: str
    cgroup_path: str

    @classmethod
    def from_line(cls, line: str) -> 'CgroupEntry':
        h_id, subsystems, path = line.strip().split(":", 2)
        return cls(
            hierarchy_id=int(h_id),
            subsystems=subsystems,
            cgroup_path=path.lstrip("/")
        )
