/*
 * ----------------------------------------------------------------------------
 * ACCELERATED convolution layer 2 (conv-accelerated LeNet workspace).
 * Same technique as conv1 / the standalone conv5x5 accelerator, but this layer
 * has CONV1_KERNELS (3) input channels, so the window is 3*5*5 = 75 taps and the
 * balanced adder tree is padded to 128.
 * ----------------------------------------------------------------------------
 */

#include "convolution2DRelu_2.h"

#ifdef __SYNTHESIS__
#include <ap_fixed.h>
typedef ap_fixed<20, 8>  data_t;
typedef ap_fixed<40, 16> acc_t;
#else
typedef float data_t;
typedef float acc_t;
#endif

#define CONV2_TAPS (CONV1_KERNELS * CONV2_KERNEL_HEIGHT * CONV2_KERNEL_WIDTH)  /* 75 */

void convolution2DRelu_2(
		const float input_kernels [CONV2_KERNELS][CONV1_KERNELS][CONV2_KERNEL_HEIGHT][CONV2_KERNEL_WIDTH],
		const float bias[CONV2_KERNELS],
		const float input_features [CONV1_KERNELS][PADDED_FEATURES_WIDTH][PADDED_FEATURES_HEIGHT],
		float output_features [CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH]
) {
	data_t in_buf[CONV1_KERNELS][PADDED_FEATURES_WIDTH][PADDED_FEATURES_HEIGHT];
	data_t w_buf[CONV2_KERNELS][CONV1_KERNELS][CONV2_KERNEL_HEIGHT][CONV2_KERNEL_WIDTH];
	acc_t  b_buf[CONV2_KERNELS];
#pragma HLS ARRAY_PARTITION variable=in_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf  complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_buf  complete dim=3
#pragma HLS ARRAY_PARTITION variable=w_buf  complete dim=4

	load_w2: for (int k = 0; k < CONV2_KERNELS; k++)
		for (int f = 0; f < CONV1_KERNELS; f++)
			for (int kh = 0; kh < CONV2_KERNEL_HEIGHT; kh++)
				for (int kw = 0; kw < CONV2_KERNEL_WIDTH; kw++)
					w_buf[k][f][kh][kw] = (data_t)input_kernels[k][f][kh][kw];

	load_b2: for (int k = 0; k < CONV2_KERNELS; k++)
		b_buf[k] = (acc_t)bias[k];

	load_in2: for (int f = 0; f < CONV1_KERNELS; f++)
		for (int h = 0; h < PADDED_FEATURES_WIDTH; h++)
			for (int w = 0; w < PADDED_FEATURES_HEIGHT; w++)
				in_buf[f][h][w] = (data_t)input_features[f][h][w];

	conv2_K: for (int k = 0; k < CONV2_KERNELS; k++) {
		conv2_FH: for (int fh = 0; fh < CONV2_CONVOLVED_FEATURE_HEIGHT; fh++) {
			conv2_FW: for (int fw = 0; fw < CONV2_CONVOLVED_FEATURE_WIDTH; fw++) {
#pragma HLS PIPELINE II=1

				acc_t products[128];
#pragma HLS ARRAY_PARTITION variable=products complete
				for (int f = 0; f < CONV1_KERNELS; f++)
					for (int kh = 0; kh < CONV2_KERNEL_HEIGHT; kh++)
						for (int kw = 0; kw < CONV2_KERNEL_WIDTH; kw++) {
#pragma HLS UNROLL
							products[f*CONV2_KERNEL_HEIGHT*CONV2_KERNEL_WIDTH + kh*CONV2_KERNEL_WIDTH + kw] =
								(acc_t)in_buf[f][fh+kh][fw+kw] * (acc_t)w_buf[k][f][kh][kw];
						}
				for (int i = CONV2_TAPS; i < 128; i++) {
#pragma HLS UNROLL
					products[i] = (acc_t)0;
				}

				for (int i = 0; i < 64; i++) {
#pragma HLS UNROLL
					products[i] = products[2*i] + products[2*i+1]; }
				for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL
					products[i] = products[2*i] + products[2*i+1]; }
				for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
					products[i] = products[2*i] + products[2*i+1]; }
				for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
					products[i] = products[2*i] + products[2*i+1]; }
				for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
					products[i] = products[2*i] + products[2*i+1]; }
				for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
					products[i] = products[2*i] + products[2*i+1]; }

				acc_t acc  = products[0] + products[1] + b_buf[k];
				acc_t zero = 0;
				output_features[k][fh][fw] = (float)((acc > zero) ? acc : zero);
			}
		}
	}
}
