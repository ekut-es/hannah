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
from hannah.models.ofa.submodules.elasticBatchnorm import ElasticWidthBatchnorm1d
from hannah.models.ofa.submodules.elastickernelconv import (
    ElasticConv1d,
    ElasticConvBn1d,
    ElasticConvBnReLu1d,
    ElasticConvReLu1d,
)
from hannah.models.ofa.submodules.elasticLinear import (
    ElasticPermissiveReLU,
    ElasticQuantWidthLinear,
    ElasticWidthLinear,
)
from hannah.models.ofa.submodules.elasticquantkernelconv import (
    ElasticQuantConv1d,
    ElasticQuantConvBn1d,
    ElasticQuantConvBnReLu1d,
    ElasticQuantConvReLu1d,
)

# A dictionary that maps the combination string of the convolution type to the class that
# implements it.
elasic_conv_classes = {
    "none": ElasticConv1d,
    "quant": ElasticQuantConv1d,
    "act": ElasticConvReLu1d,
    "actquant": ElasticQuantConvReLu1d,
    "norm": ElasticConvBn1d,
    "normquant": ElasticQuantConvBn1d,
    "normact": ElasticConvBnReLu1d,
    "normactquant": ElasticQuantConvBnReLu1d,
}


# A tuple of all the classes that are subclasses of `ElasticBaseConv`.
elastic_conv_type = (
    ElasticConv1d,
    ElasticConvReLu1d,
    ElasticConvBn1d,
    ElasticConvBnReLu1d,
    ElasticQuantConv1d,
    ElasticQuantConvReLu1d,
    ElasticQuantConvBn1d,
    ElasticQuantConvBnReLu1d,
)

elastic_forward_type = (
    ElasticConv1d,
    ElasticConvReLu1d,
    ElasticConvBn1d,
    ElasticConvBnReLu1d,
    ElasticQuantConv1d,
    ElasticQuantConvReLu1d,
    ElasticQuantConvBn1d,
    ElasticQuantConvBnReLu1d,
    ElasticWidthLinear,
    ElasticQuantWidthLinear,
)

elastic_Linear_type = (
    ElasticWidthLinear,
    ElasticQuantWidthLinear,
)

elastic_all_type = (
    ElasticConv1d,
    ElasticConvReLu1d,
    ElasticConvBn1d,
    ElasticConvBnReLu1d,
    ElasticQuantConv1d,
    ElasticQuantConvReLu1d,
    ElasticQuantConvBn1d,
    ElasticQuantConvBnReLu1d,
    ElasticWidthBatchnorm1d,
    ElasticWidthLinear,
    ElasticQuantWidthLinear,
    ElasticPermissiveReLU,
)
