from typing import Optional


class MemoryType:
    def __init__(self, size: Optional[int] = None, name: Optional[str] = "") -> None:
        self.size = size
        self.name = name
