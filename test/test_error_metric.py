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
import pytest
import torch
from torchmetrics import Accuracy

from hannah.modules.metrics import BinaryError, Error, MulticlassError, MultilabelError


@pytest.mark.parametrize(
    "task,predictions,labels,n,expect",
    [
        (
            "binary",
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
            2,
            0.0,
        ),
        ("binary", [0, 0], [0, 1], 2, 0.5),
        ("binary", [1, 0], [0, 1], 2, 1.0),
        ("multiclass", [[0, 1], [0, 1]], [[0, 1], [0, 1]], 2, 0.0),
        ("multiclass", [[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]], 3, 0.0),
        ("multiclass", [[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]], 3, 0.0),
        ("multilabel", [[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]], 3, 0.0),
        ("multilabel", [[0, 1, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 1]], 3, 0.0),
        ("multilabel", [[0, 1, 1], [0, 1, 1]], [[1, 0, 0], [1, 0, 0]], 3, 1.0),
    ],
)
def test_error(task, predictions, labels, n, expect):
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    if task == "binary":
        task_error = BinaryError()
        error = Error(task)
        accuracy = Accuracy(task)
    elif task == "multiclass":
        task_error = BinaryError(num_classes=n)
        error = Error(task, num_classes=n)
        accuracy = Accuracy(task, num_classes=n)
    elif task == "multilabel":
        task_error = BinaryError(num_labels=n)
        error = Error(task, num_labels=n)
        accuracy = Accuracy(task, num_labels=n)

    task_err = task_error(predictions, labels)
    err = error(predictions, labels)
    acc = accuracy(predictions, labels)

    assert task_err == err
    assert 1.0 - acc == err
    assert 1.0 - acc == task_err
    assert task_err == expect
    assert err == expect
