from hannah.models.ofa.submodules.elastickernelconv import ElasticConv1d, \
    ElasticConvBn1d, ElasticConvBnReLu1d
from hannah.models.ofa.submodules.elasticquantkernelconv import \
    ElasticQuantConv1d, ElasticQuantConvBn1d, ElasticQuantConvBnReLu1d
from hannah.models.ofa.submodules.elasticwidthmodules import \
    ElasticWidthBatchnorm1d, ElasticWidthLinear, ElasticPermissiveReLU

elastic_conv_type = (
                ElasticConv1d,
                ElasticConvBn1d,
                ElasticConvBnReLu1d,
                ElasticQuantConv1d,
                ElasticQuantConvBn1d,
                ElasticQuantConvBnReLu1d,
            )

elastic_forward_type = (
                ElasticConv1d,
                ElasticConvBn1d,
                ElasticConvBnReLu1d,
                ElasticQuantConv1d,
                ElasticQuantConvBn1d,
                ElasticQuantConvBnReLu1d,
                ElasticWidthLinear,
            )

elastic_all_type = (
                ElasticConv1d,
                ElasticConvBn1d,
                ElasticConvBnReLu1d,
                ElasticQuantConv1d,
                ElasticQuantConvBn1d,
                ElasticQuantConvBnReLu1d,
                ElasticWidthBatchnorm1d,
                ElasticWidthLinear,
                ElasticPermissiveReLU,
            )
