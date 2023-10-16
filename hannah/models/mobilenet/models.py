from hannah.nas.functional_operators.op import scope
from hannah.models.mobilenet.operators import batch_norm, inverted_residual, conv2d, linear, pointwise_conv2d, adaptive_avg_pooling, relu


@scope
def mobilenetv2(name, input, num_classes=10):
    cfgs = [  # t, c, n, s
        #    [1,  16, 1, 1],
        #    [6,  24, 1, 1],
        #    [6,  32, 1, 1],
        #    [6,  64, 1, 2],
           [6,  96, 1, 2],
           [6, 160, 1, 2],
           [6, 320, 1, 2],]
    out = conv2d(input, out_channels=32, kernel_size=3, stride=2)
    out = batch_norm(out)
    out = relu(out)
    for t, c, n, s in cfgs:
        for i in range(n):
            out = inverted_residual(out, c, s if i == 0 else 1, t)
    out = pointwise_conv2d(out, out_channels=1280)
    out = batch_norm(out)
    out = relu(out)
    out = adaptive_avg_pooling(out)
    out = linear(out, num_classes)
    return out
