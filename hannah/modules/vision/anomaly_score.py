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


"""class AnomalyScore(CatMetric):
    def __init__(self, percentile, nan_strategy="warn", **kwargs):
        super().__init__(nan_strategy=nan_strategy, **kwargs)
        self.percentile = percentile

    def compute(self):
        anomaly_score = None
        train_losses = super().compute()
        if train_losses:
            normalized_train_errors = torch.stack(train_losses) / (
                torch.max(torch.stack(train_losses), dim=0).values
            )
            anomaly_score = np.percentile(
                normalized_train_errors.cpu().numpy(), self.percentile
            )
        return anomaly_score"""
