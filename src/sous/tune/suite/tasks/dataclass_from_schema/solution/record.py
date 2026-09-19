import dataclasses
from dataclasses import dataclass, field

_REQUIRED = ("id", "name", "active")


@dataclass
class Record:
    id: int
    name: str
    active: bool
    tags: list[str] = field(default_factory=list)
    score: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Record:
        missing = [key for key in _REQUIRED if key not in data]
        if missing:
            raise ValueError(f"missing required keys: {', '.join(missing)}")
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            active=bool(data["active"]),
            tags=list(data.get("tags", [])),
            score=None if data.get("score") is None else float(data["score"]),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
