"""Data definitions kept independent from any particular dataset."""

from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = ("can", "plastic_bottle", "paper", "other")


@dataclass(frozen=True)
class RecyclingExample:
    """One image and its object-level class label."""

    image_path: Path
    class_id: int

    def __post_init__(self) -> None:
        if not 0 <= self.class_id < len(CLASS_NAMES):
            raise ValueError(f"class_id must be between 0 and {len(CLASS_NAMES) - 1}")

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]
