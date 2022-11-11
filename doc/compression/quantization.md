<!--
Copyright (c) 2022 University of Tübingen.

This file is part of hannah.
See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.

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
# Quantization

The training framework provides quantization aware training support for model factory based models.

We support the following quantization schemes.

1. Symmetric linear Quantization:

Symmetric linear quantization applies the following transformation on a floating point value $v$:

$\mbox{quantized}(v)\ =\ \mbox{clamp}(\mbox{round}(v\cdot2^{\mbox{bits}-1}), 2^{-\mbox{bits}-1}, 2^{\mbox{bits}-1}-1)$

2. Power of 2 Quantization with Zero:

Power of 2 quantization applies the following transformation to a floating point value v:

\(
\mbox{quantized}(v)\ =\
   \begin{cases}
   \mbox{sign}(v)\ \cdot\ 2^{\mbox{round}(\log2(\mbox{abs}(v)))} & \text{if}\ v\ \ge\ 2^{2^{\text{bits}}\\
   0                                                             & \text{otherwise} \\
   \end{cases}
\)

Power of 2 Quantization can currently only be enabled for weights.


## Noisy Quantization

As described in https://openreview.net/pdf?id=dV19Yyi1fS3 quantizing only a subset of model weights during training
can improve accuracies of quantized networks compared to full quantization aware training.

The probability of quantizing a weight can be given as parameter noise_prob in in the qconfig.
Unfortunately this introduces an additional hyperparameter in the quantization space, good values
for noise prob seem to be in the range of 0.7 to 0.9.

Even when using noisy quantization eval and test are always run on fully quantized networks.

## Configuration

Quantization can be configured as the qconfig attribute of factory based models:

qconfig:
  _target_: speech_recognition.models.factory.qconfig.get_trax_qat_qconfig
  config:
    bw_b: 8  # Bitwidth for bias
    bw_w: 6  # Bitwidth for weights
    bw_f: 8  # Bitwidth for activations
    power_of_2: false  # Use power of two quantization for weights
    noise_prob: 0.7    # Probability of quantizing a value during training

Aditionally standard pytorch quantization aware training is supported by using a standard pytorch qconfig.

    qconfig:
        _target_: torch.quantization.get_default_qat_qconfig
        backend: fbgemm

In this case no quantization noise is supported.
