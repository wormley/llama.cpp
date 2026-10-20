#ifndef GGML_SYCL_DSV4_HC_HPP
#define GGML_SYCL_DSV4_HC_HPP

#include "common.hpp"

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);

#endif // GGML_SYCL_DSV4_HC_HPP
