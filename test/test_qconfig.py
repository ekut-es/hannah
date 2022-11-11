#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
