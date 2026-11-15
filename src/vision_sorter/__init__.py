"""Foundation package for the vision-guided recycling sorter."""

from .data import CLASS_NAMES, RecyclingExample
from .model import TinySorterCNN

__all__ = ["CLASS_NAMES", "RecyclingExample", "TinySorterCNN"]
