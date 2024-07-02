from hannah.nas.functional_operators.op import scope
from hannah.models.conv_vit.operators import conv2d, self_attention2d, relu_linear_attention


@scope
def attention2d(input, num_heads, d_model):
    # [B, C, H, W] --> 3 tensors each of shape [B, h*d, H, W]
    inner_dim = num_heads * d_model
    q = q_proj(input, inner_dim)
    k = k_proj(input, inner_dim)
    v = v_proj(input, inner_dim)

    # 3 x [B, h*d, H, W] --> [B, h*d, H, W]
    out = self_attention2d(q, k, v, num_heads=num_heads, d_model=d_model)

    return out


def relu_lin_attention(input, num_heads, d_model):
    # [B, C, H, W] --> 3 tensors each of shape [B, h*d, H, W]
    inner_dim = num_heads * d_model
    q = q_proj(input, inner_dim)
    k = k_proj(input, inner_dim)
    v = v_proj(input, inner_dim)

    # 3 x [B, h*d, H, W] --> [B, h*d, H, W]
    out = relu_linear_attention(q, k, v, num_heads=num_heads, d_model=d_model)

    return out


@scope
def q_proj(input, out_dim):
    q = conv2d(input, out_dim, kernel_size=1)
    return q


@scope
def k_proj(input, out_dim):
    k = conv2d(input, out_dim, kernel_size=1)
    return k


@scope
def v_proj(input, out_dim):
    v = conv2d(input, out_dim, kernel_size=1)
    return v
