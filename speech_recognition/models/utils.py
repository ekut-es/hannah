from enum import Enum

import torch
import torch.nn as nn

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def save_onnx(self, filename, dummy_input):
        with torch.onnx.set_training(self, False):
            torch.onnx.export(self, dummy_input, filename) 


class ConfigType(Enum):
    #Models from honk.ai: https://github.com/castorini/Honk
    # These models use mel features
    HONK_CNN_TRAD_POOL2   = "honk-cnn-trad-pool2" # default full model (TF variant)
    HONK_CNN_ONE_STRIDE1  = "honk-cnn-one-stride1" # default compact model (TF variant)
    HONK_CNN_ONE_FPOOL3   = "honk-cnn-one-fpool3"
    HONK_CNN_ONE_FSTRIDE4 = "honk-cnn-one-fstride4"
    HONK_CNN_ONE_FSTRIDE8 = "honk-cnn-one-fstride8"
    HONK_CNN_TPOOL2       = "honk-cnn-tpool2"
    HONK_CNN_TPOOL3       = "honk-cnn-tpool3"
    HONK_CNN_TSTRIDE2     = "honk-cnn-tstride2"
    HONK_CNN_TSTRIDE4     = "honk-cnn-tstride4"
    HONK_CNN_TSTRIDE8     = "honk-cnn-tstride8"
    HONK_RES15            = "honk-res15"
    HONK_RES26            = "honk-res26"
    HONK_RES8             = "honk-res8"
    HONK_RES15_NARROW     = "honk-res15-narrow"
    HONK_RES8_NARROW      = "honk-res8-narrow"
    HONK_RES26_NARROW     = "honk-res26-narrow"
    
    # These models use raw audio
    EKUT_RAW_CNN1          = "ekut-raw-cnn1"
    EKUT_RAW_CNN2          = "ekut-raw-cnn2"
    EKUT_RAW_CNN3          = "ekut-raw-cnn3"
    EKUT_RAW_CNN4          = "ekut-raw-cnn4"
    EKUT_RAW_CNN1_RELU     = "ekut-raw-cnn1-relu"
    EKUT_RAW_CNN2_RELU     = "ekut-raw-cnn2-relu"
    EKUT_RAW_CNN3_RELU     = "ekut-raw-cnn3-relu"
    EKUT_RAW_CNN4_RELU     = "ekut-raw-cnn4-relu"
    EKUT_RAW_CNN5_RELU     = "ekut-raw-cnn5-relu"
    EKUT_RAW_INVERTED_RES1 = "ekut-raw-inverted-res1"
    
    # Models for Hello Edge
    HELLO_DNN_SMALL = "hello-dnn-small"
    HELLO_DNN_MEDIUM = "hello-dnn-medium"
    HELLO_DNN_LARGE = "hello-dnn-large"
    HELLO_DS_CNN_SMALL = "hello-ds-cnn-small"
    HELLO_DS_CNN_MEDIUM = "hello-ds-cnn-medium"
    HELLO_DS_CNN_LARGE = "hello-ds-cnn-large"
