from hannah.models.ofa.submodules.elastickernelconv import (
    ElasticConv1d,
    ElasticConvBn1d,
    ElasticConvBnReLu1d,
    ElasticConvReLu1d,
)

from hannah.models.ofa.submodules.elasticquantkernelconv import (
    ElasticQuantConv1d,
    ElasticQuantConvBn1d,
    ElasticQuantConvBnReLu1d,
    ElasticQuantConvReLu1d,
)
from hannah.models.ofa.submodules.elasticwidthmodules import (
    ElasticWidthBatchnorm1d,
    ElasticWidthLinear,
    ElasticPermissiveReLU,
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
    ElasticPermissiveReLU,
)
