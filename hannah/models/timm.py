import logging
import math
from typing import Any, Mapping, Tuple, Union

import timm
import torch
import torch.nn as nn
from torch import nn

import hydra

logger = logging.getLogger(__name__)


class DefaultClassifierHead(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()


class DefaultDecoderHead(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()

        input_x = input_dim[-2]
        input_y = input_dim[-1]
        input_channels = input_dim[-3]

        latent_x = latent_dim[-2]
        latent_y = latent_dim[-1]

        scale_factor_x = math.log2(input_x / latent_x)
        scale_factor_y = math.log2(input_y / latent_y)

        print(scale_factor_x, input_x, latent_x)

        stages = int(round((max(scale_factor_x, scale_factor_y))))

        logger.info("Using %d autoencoder stages", stages)

        upscale = []
        channels = latent_dim[-3]
        dim_x = latent_dim[-2]
        dim_y = latent_dim[-1]
        for stage_num in range(stages):
            stage = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=(1, 1)),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(2.0),
                nn.Upsample(scale_factor=2.0),
            )

            dim_x *= 2.0
            dim_y *= 2.0

            upscale.append(stage)

        stage = nn.Sequential(
            nn.Conv2d(channels, input_channels, 3, padding=(1, 1)),
            nn.BatchNorm2d(input_channels),
            nn.Upsample(size=(input_x, input_y)),
        )
        upscale.append(stage)

        self.layers = nn.Sequential(*upscale)

    def forward(self, x):
        return self.layers(x)


class TimmModel(nn.Module):
    def __init__(
        self,
        name: str,
        input_shape: Tuple[int, int, int],
        pretrained: bool = True,
        decoder: Union[Mapping[str, Any], bool] = True,
        classifier: Union[Mapping[str, Any], bool] = True,
        **kwargs
    ):
        super().__init__()
        self.name = name

        dummy_input = torch.randn(input_shape)

        self.encoder = timm.create_model(
            name, num_classes=0, global_pool="", pretrained=pretrained, **kwargs
        )

        dummy_latent = self.encoder(dummy_input)

        self.decoder = None
        if decoder is True:
            self.decoder = DefaultDecoderHead(dummy_latent.shape, input_shape)
        elif decoder:
            decoder = hydra.utils.instantiate(
                decoder, input_shape=input_shape, latent_shape=dummy_latent.shape
            )

        self.classifier = None
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        result = {}

        latent = self.encoder(x)
        result["latent"] = latent

        if self.decoder is not None:
            decoded = self.decoder(latent)
            result["decoded"] = decoded

        if self.classifier is not None:
            pred = self.classifier(latent)
            result["logits"] = pred

        return result
