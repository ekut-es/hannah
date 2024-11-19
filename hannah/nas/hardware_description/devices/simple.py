from ..device import Device

class SimpleDevice(Device):
    name = "simple_hwa"
    description = "A simple Abstract Hardware Device with Conv2d -> ReLU acceleration and configurable precision"
    
    def __init__(self, precision=8):
        super().__init__()
        self.precision = precision
   
     