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
import matplotlib.pyplot as plt
import seaborn as sns
import torch


class LogActDistributionHook:
    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = 0

    def __call__(self, _module, input, output):
        with torch.no_grad():
            if self.sum is None:
                self.sum = torch.clone(output)
            else:
                self.sum += output
            self.count += 1

        return output


_distribution_hooks = []


def log_distribution_install_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.ReLU)
            or isinstance(module, torch.nn.BatchNorm2d)
        ):
            hook = LogActDistributionHook(name)
            handle = module.register_forward_hook(hook)
            hooks.append((handle, hook))

    return hooks


def log_distribution_plot(hooks):
    for handle, hook in hooks:
        handle.remove()

        data = hook.sum / hook.count
        data = data.detach().cpu().numpy()

        plot = sns.displot(data.flatten(), kind="hist", bins=20)
        plot.savefig(f"act_distribution_{hook.name}.png")
        plt.clf()
