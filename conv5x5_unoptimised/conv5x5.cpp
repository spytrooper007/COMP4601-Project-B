#include "conv5x5.h"

/*
 * Hardware 5x5 convolution + ReLU kernel for Vitis HLS -- UNOPTIMISED BASELINE.
 *
 * No performance pragmas (no PIPELINE / UNROLL / ARRAY_PARTITION): the six-deep
 * loop nest runs sequentially, reusing a single multiply-add unit. This is the
 * baseline that the optimised version is measured against.
 *
 * Loop structure (the "six deep loop nest" from the project slides):
 *   OC  - output channel / kernel
 *   OH  - output row
 *   OW  - output col
 *     IC/KH/KW           - the 5x5 window (25 MACs), evaluated one at a time
 *
 * Inputs/weights are still copied once into on-chip BRAM so each input pixel is
 * reused across the sliding window instead of re-fetched over AXI.
 */
void conv5x5(const float *input,
             const float *weights,
             const float *bias,
             float       *output) {

#pragma HLS INTERFACE m_axi     port=input   bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi     port=weights bundle=gmem1 depth=150
#pragma HLS INTERFACE m_axi     port=bias    bundle=gmem1 depth=6
#pragma HLS INTERFACE m_axi     port=output  bundle=gmem2 depth=4704
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    /* On-chip copies of all operands. */
    float in_buf[IN_CH][IN_H][IN_W];
    float w_buf[OUT_CH][IN_CH][K][K];
    float b_buf[OUT_CH];

    /* Load weights into BRAM. */
    load_w: for (int oc = 0; oc < OUT_CH; oc++)
        for (int ic = 0; ic < IN_CH; ic++)
            for (int kh = 0; kh < K; kh++)
                for (int kw = 0; kw < K; kw++)
                    w_buf[oc][ic][kh][kw] =
                        weights[((oc * IN_CH + ic) * K + kh) * K + kw];

    load_b: for (int oc = 0; oc < OUT_CH; oc++)
        b_buf[oc] = bias[oc];

    load_in: for (int ic = 0; ic < IN_CH; ic++)
        for (int h = 0; h < IN_H; h++)
            for (int w = 0; w < IN_W; w++)
                in_buf[ic][h][w] = input[(ic * IN_H + h) * IN_W + w];

    /* Convolution. */
    conv_OC: for (int oc = 0; oc < OUT_CH; oc++) {
        conv_OH: for (int oh = 0; oh < OUT_H; oh++) {
            conv_OW: for (int ow = 0; ow < OUT_W; ow++) {
                float acc = 0.0f;
                conv_IC: for (int ic = 0; ic < IN_CH; ic++) {
                    conv_KH: for (int kh = 0; kh < K; kh++) {
                        conv_KW: for (int kw = 0; kw < K; kw++) {
                            acc += in_buf[ic][oh + kh][ow + kw] *
                                   w_buf[oc][ic][kh][kw];
                        }
                    }
                }
                float res = acc + b_buf[oc];
                /* ReLU */
                output[(oc * OUT_H + oh) * OUT_W + ow] = (res > 0.0f) ? res : 0.0f;
            }
        }
    }
}
