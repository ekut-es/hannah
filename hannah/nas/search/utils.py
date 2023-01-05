from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class WorklistItem:
    parameters: Any
    results: Dict[str, float]
