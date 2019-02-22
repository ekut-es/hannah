from . import honk
from . import ekut
from . import hello

from .utils import ConfigType
    
def find_model(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if conf.startswith("honk-res"):
        return honk.SpeechResModel
    elif conf.startswith("honk-cnn"):
        return honk.SpeechModel
    elif conf.startswith("ekut-raw-inv-res"):
        return ekut.RawSpeechModelInvertedResidual
    elif conf.startswith("ekut-raw-cnn"):
        return ekut.RawSpeechModel
    elif conf.startswith("hello-dnn"):
        return hello.DNNSpeechModel
    elif conf.startswith("hello-ds-cnn"):
        return hello.DSCNNSpeechModel
    
    raise Exception("Could not find model for {}".format(str(conf)))
    

def find_config(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if(conf.startswith("honk")):
        return honk.configs[conf]
    elif conf.startswith("ekut"):
        return ekut.configs[conf]
    elif conf.startswith("hello"):
        return hello.configs[conf]
    
    raise Exception("Could not find config for {}".format(str(conf)))
