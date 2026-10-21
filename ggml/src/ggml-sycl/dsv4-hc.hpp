#ifndef GGML_SYCL_DSV4_HC_HPP
#define GGML_SYCL_DSV4_HC_HPP

#include "common.hpp"

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the nodes of a fused gate chain feeding src[2] — either the
// SCALE -> SIGMOID -> SCALE chain of qwen4exp (scalar gate) or the mul(scale) + add(bias) +
// sigmoid + scale chain of glm5-next/deepseek4 (affine gate; gate_bias is the ADD's bias operand)
void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr,
        const ggml_tensor * gate_bias = nullptr);

#endif // GGML_SYCL_DSV4_HC_HPP
