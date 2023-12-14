#
# Copyright (c) 2023 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
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

DVCLIVE_AVAILABLE = True
try:
    from dvclive.lightning import DVCLiveLogger
except ImportError:
    DVCLIVE_AVAILABLE = False
    DVCLiveLogger = object


from lightning.fabric.utilities.logger import _convert_params, _sanitize_callable_params
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf


class DVCLogger(DVCLiveLogger):
    def __init__(self, *args, **kwargs):
        if not DVCLIVE_AVAILABLE:
            raise ImportError(
                "dvclive is not installed. Please install it with `pip install 'dvclive[lightning]'"
            )

        super().__init__(*args, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        params = _convert_params(params)
        params = _sanitize_callable_params(params)

        params = OmegaConf.to_container(OmegaConf.create(params))
        self.experiment.log_params(params)
