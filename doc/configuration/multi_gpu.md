<!--
Copyright (c) 2023 Hannah contributors.

This file is part of hannah.
See https://github.com/ekut-es/hannah for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Multi GPU training

Hannah supports multi GPU-Training using the lightning distributed APIs:

We provide preset trainer configs for distributed data parallel training:

```hannah-train trainer=ddp trainer.devices=[0,1]```


And for sharded training using fairscale:


```hannah-train trainer=sharded trainer.devices=[0,1]```


Sharded training distributes some of the model parameters across multiple devices and allows fitting bigger models in the same amount of GPU memory.
