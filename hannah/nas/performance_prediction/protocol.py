#
# Copyright (c) 2024 Hannah contributors.
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
from typing import (
    TYPE_CHECKING,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from hannah.modules.base import ClassifierModule

if TYPE_CHECKING:
    import torch

    from ..functional_operators.op import Tensor


InputShape = Union["Tensor", Tuple[int, ...], "torch.tensor"]


@runtime_checkable
class Predictor(Protocol):
    def predict(
        self, model: ClassifierModule, input: Optional[InputShape] = None
    ) -> Mapping[str, float]:
        """Pedicts performance metrisc  of a model.

        Performance metrics are returned as a dictionary with the metric name as key and the metric value as floating point value.

        Args:
            model (ClassifierModule): The model to predict the performance of.
            input (_type_, optional): Input shape of input  . Defaults to None.
        """
        ...


class FitablePredictor(Predictor):
    def load(self, result_folder: str):
        """Load predefined model from a folder.

        Args:
            result_folder (str): Path to the folder containing the model or training data to recreate the model.
        """
        ...

    def update(self, new_data, input=None):
        """Update the model with new data."""
        ...
