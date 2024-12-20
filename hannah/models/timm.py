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

import logging
import math
from collections import namedtuple
from typing import Any, Mapping, Tuple, Union

import hydra
import timm
import torch
import torch.nn as nn

from ._vendor import resnet_mc_dropout

logger = logging.getLogger(__name__)


class DefaultAnomalyDetector(nn.Module):
    """ """

    def __init__(self, latent_shape):
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(1)

    def forward(self, x):
        """

        Simple anomaly detection head for a neural network

        Args:
          x: Tensor of logits

        Returns: A single element floating point tensor representing the anomaly score

        """
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = nn.functional.sigmoid(x)

        return x


class DefaultClassifierHead(nn.Module):
    """ """

    def __init__(self, latent_shape, num_classes):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1)) if len(latent_shape) == 4 else nn.Identity()
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x (torch.Tensor):

        Returns:
          Resulting torch.Tensor after applying classification
        """
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class DefaultProjectionHead(nn.Module):
    """Default projection head for semi supervised classification learning"""

    def __init__(self, latent_shape, hidden_dim, output_dim):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.LazyLinear(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for default Projection Head

        Args:
          x: Input tensor

        Returns:
            output tensor
        """
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class DefaultDecoderHead(nn.Module):
    def __init__(self, latent_shape, input_shape):
        """Default Decoder Head for autoencoders using TransposedConv2D

        Args:
        latent_shape(Tuple): Shape (CxHxW) of the latent representation of the autoencoder
        input_shape(Tuple): Shape (CxHxW) of the reconstructed image
        """
        super().__init__()

        input_x = input_shape[-2]
        input_y = input_shape[-1]
        input_channels = input_shape[-3]

        latent_x = latent_shape[-2]
        latent_y = latent_shape[-1]

        scale_factor_x = math.log2(input_x / latent_x)
        scale_factor_y = math.log2(input_y / latent_y)

        print(scale_factor_x, input_x, latent_x)

        stages = int(round((max(scale_factor_x, scale_factor_y))))

        logger.info("Using %d autoencoder stages", stages)

        upscale = []
        channels = latent_shape[-3]
        dim_x = latent_shape[-2]
        dim_y = latent_shape[-1]
        for stage_num in range(stages):
            out_channels = channels // 2
            stage = nn.Sequential(
                nn.ConvTranspose2d(
                    channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=(1, 1),
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(2.0),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(2.0),
            )

            dim_x *= 2.0
            dim_y *= 2.0
            channels = out_channels

            upscale.append(stage)

        stage = nn.Sequential(
            nn.Upsample(size=(input_x, input_y)),
            nn.Conv2d(channels, input_channels, 3, padding=(1, 1)),
            nn.BatchNorm2d(input_channels),
            nn.Tanh(),
        )
        upscale.append(stage)

        self.layers = nn.Sequential(*upscale)

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        return self.layers(x)


ModelResult = namedtuple("ModelResult", ["latent", "decoded", "projection", "logits"])


class TimmModel(nn.Module):
    """ """

    def __init__(
        self,
        name: str,
        input_shape: Tuple[int, int, int],
        pretrained: bool = False,
        decoder: Union[Mapping[str, Any], bool] = False,
        classifier: Union[Mapping[str, Any], bool] = True,
        projector: Union[Mapping[str, Any], bool] = False,
        stem: str = "auto",
        labels: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.name = name

        dummy_input = torch.randn(tuple(input_shape))

        self.encoder = timm.create_model(name, num_classes=0, global_pool="", pretrained=pretrained, **kwargs)
        _, input_channels, input_x, input_y = dummy_input.shape
        if stem == "auto":
            logger.info("""Using default logger for automatic stem creation""")
            if input_channels != 3 or input_x < 160 or input_y < 160:
                logger.info("Small input size detected trying to adopt small input size")
                if hasattr(self.encoder, "conv1"):
                    input_conv = self.encoder.conv1
                    out_channels = input_conv.out_channels
                    new_conv = torch.nn.Conv2d(input_channels, out_channels, 3, 1, padding=1)
                    self.encoder.conv1 = new_conv
                elif hasattr(self.encoder, "conv_stem"):
                    input_conv = self.encoder.conv_stem
                    out_channels = input_conv.out_channels
                    new_conv = torch.nn.Conv2d(input_channels, out_channels, 3, 1, padding=1)
                    self.encoder.conv_stem = new_conv
                else:
                    logger.critical(
                        "Selected timm model does not have any known conv attr, cannot adopt input convolution for automatic stem selection"
                    )

                if hasattr(self.encoder, "maxpool"):
                    self.encoder.maxpool = torch.nn.Identity()

        elif stem == "default":
            logger.info("""Using default stem for pulp model""")
        else:
            logger.critical("""%s could not find stem for input element""", stem)

        with torch.no_grad():
            self.encoder.eval()
            dummy_latent = self.encoder(dummy_input)
            self.encoder.train()

        self.decoder = None
        if decoder is True:
            self.decoder = DefaultDecoderHead(dummy_latent.shape, input_shape)
        elif decoder:
            decoder = hydra.utils.instantiate(decoder, input_shape=input_shape, latent_shape=dummy_latent.shape)

        self.classifier = None
        if labels > 0:
            if classifier is True:
                self.classifier = DefaultClassifierHead(latent_shape=dummy_latent.shape, num_classes=labels)
            elif classifier:
                self.classifier = hydra.utils.instantiate(latent_shape=dummy_latent.shape, num_classes=labels)

        self.projector = None
        if projector is True:
            self.projector = DefaultProjectionHead(dummy_latent.shape, 1024, 256)
        elif projector:
            self.projector = hydra.utils.instantiate(projector, latent_shape=dummy_latent.shape)

        self.input_shape = input_shape

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
          x: torch.Tensor:
          x: torch.Tensor:

        Returns:

        """

        latent = self.encoder(x)

        decoded = torch.tensor([])
        if self.decoder is not None:
            decoded = self.decoder(latent)

        logits = torch.tensor([])
        if self.classifier is not None:
            logits = self.classifier(latent)

        projection = torch.tensor([])
        if self.projector is not None:
            projection = self.projector(latent)

        result = ModelResult(latent=latent, decoded=decoded, projection=projection, logits=logits)

        return result
