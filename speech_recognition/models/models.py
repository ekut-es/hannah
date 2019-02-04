from . import honk
from . import ekut

from .utils import ConfigType
    
def find_model(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if conf.startswith("honk-res"):
        return honk.SpeechResModel
    elif conf.startswith("honk-cnn"):
        return honk.SpeechModel
    elif conf.startswith("ekut"):
        return ekut.RawSpeechModel
    
    raise Exception("Could not find model for {}".format(str(conf)))
    

def find_config(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if(conf.startswith("honk")):
        return honk.configs[conf]
    elif conf.startswith("ekut"):
        return ekut.configs[conf]
    
    raise Exception("Could not find config for {}".format(str(conf)))
