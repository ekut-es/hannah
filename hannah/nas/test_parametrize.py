from typing import Optional
from hannah.nas.parameters import parametrize, IntScalarParameter


@parametrize
class Test:
    def __init__(self, a: int, b: Optional[str] = None, c: str = "test"):
        self.a = a ** 2
        self.b = b
        self.c = c


test = Test(IntScalarParameter(min=10, max=20))

breakpoint()
