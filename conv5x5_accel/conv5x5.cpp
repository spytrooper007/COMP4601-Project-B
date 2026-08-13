#include "conv5x5.h"

void conv5x5(const float *input,
             const float *weights,
             const float *bias,
             float       *output) {

#pragma HLS INTERFACE m_axi     port=input   bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi     port=weights bundle=gmem1 depth=150
#pragma HLS INTERFACE m_axi     port=bias    bundle=gmem1 depth=6
#pragma HLS INTERFACE m_axi     port=output  bundle=gmem2 depth=4704
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    data_t in_buf[IN_CH][IN_H][IN_W];
    data_t w_buf[OUT_CH][IN_CH][K][K];
    acc_t  b_buf[OUT_CH];

#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=3
#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=4
#pragma HLS ARRAY_PARTITION variable=in_buf complete dim=1

    load_w: for (int oc = 0; oc < OUT_CH; oc++)
        for (int ic = 0; ic < IN_CH; ic++)
            for (int kh = 0; kh < K; kh++)
#pragma HLS PIPELINE II=1
                for (int kw = 0; kw < K; kw++)
                    w_buf[oc][ic][kh][kw] =
                        (data_t)weights[((oc * IN_CH + ic) * K + kh) * K + kw];

    load_b: for (int oc = 0; oc < OUT_CH; oc++)
#pragma HLS PIPELINE II=1
        b_buf[oc] = (acc_t)bias[oc];

    load_in: for (int ic = 0; ic < IN_CH; ic++)
        for (int h = 0; h < IN_H; h++)
#pragma HLS PIPELINE II=1
            for (int w = 0; w < IN_W; w++)
                in_buf[ic][h][w] = (data_t)input[(ic * IN_H + h) * IN_W + w];

    conv_OC: for (int oc = 0; oc < OUT_CH; oc++) {
        conv_OH: for (int oh = 0; oh < OUT_H; oh++) {
            conv_OW: for (int ow = 0; ow < OUT_W; ow++) {
#pragma HLS PIPELINE II=1

                acc_t products[K * K];
#pragma HLS ARRAY_PARTITION variable=products complete

                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
#pragma HLS UNROLL
                        products[kh * K + kw] =
                            (acc_t)in_buf[0][oh + kh][ow + kw] *
                            (acc_t)w_buf[oc][0][kh][kw];
                    }
                }

                acc_t tree[32];
#pragma HLS ARRAY_PARTITION variable=tree complete

                for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
                    tree[i] = (i < K * K) ? products[i] : (acc_t)0;
                }

                for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                    tree[i] = tree[2*i] + tree[2*i+1];
                }
                for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
                    tree[i] = tree[2*i] + tree[2*i+1];
                }
                for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
                    tree[i] = tree[2*i] + tree[2*i+1];
                }
                for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
                    tree[i] = tree[2*i] + tree[2*i+1];
                }
                acc_t acc = tree[0] + tree[1];

                acc += b_buf[oc];

                acc_t zero = 0;
                acc_t res = (acc > zero) ? acc : zero;
                output[(oc * OUT_H + oh) * OUT_W + ow] = (float)res;
            }
        }
    }
}
