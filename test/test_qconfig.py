import copy
import copyreg
import pickle

from omegaconf import DictConfig

from hannah.models.factory.qconfig import get_trax_qat_qconfig

config = DictConfig(content={"bw_w": 6, "bw_f": 8, "bw_b": 8})


def test_picklable():
    qconfig = get_trax_qat_qconfig(config)

    copied = copy.deepcopy(qconfig)

    pickled = pickle.dumps(qconfig)
    unpickled = pickle.loads(pickled)

    act = unpickled.activation()
    assert act.bits == 8
    assert act.noise_prob == 1.0

    act = unpickled.activation()
    assert act.bits == 8
    assert act.noise_prob == 1.0
