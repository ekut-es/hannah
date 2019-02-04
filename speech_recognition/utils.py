import torch
import torch.nn as nn

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def export_onnx(self, filename):
        torch.onnx.export(self, dummy_input,filename, verbose=True) 
