/*
 * ----------------------------------------------------------------------------
 * ACCELERATED convolution layer 1.
 * Same technique as the standalone conv5x5 accelerator:
 *   - fixed-point datapath (ap_fixed) during synthesis; float during C-sim
 *   - inputs/weights copied into on-chip BRAM
 *   - output-pixel loop pipelined (II=1)
 *   - the 5x5 window fully unrolled, products summed via a balanced adder tree
 * Only the compute is changed; the float interface is identical to the baseline.
 * ----------------------------------------------------------------------------
 */

#include "convolution2DRelu_1.h"

#ifdef __SYNTHESIS__
#include <ap_fixed.h>
typedef ap_fixed<20, 8>  data_t;
typedef ap_fixed<40, 16> acc_t;
#else
typedef float data_t;
typedef float acc_t;
#endif

#define CONV1_TAPS (CONV1_FEATURES * CONV1_KERNEL_HEIGHT * CONV1_KERNEL_WIDTH)  /* 25 */

void convolution2DRelu_1(
		const float input_kernels [CONV1_KERNELS][CONV1_FEATURES][CONV1_KERNEL_HEIGHT][CONV1_KERNEL_WIDTH],
		const float bias[CONV1_KERNELS],
		const float input_features [CONV1_FEATURES][CONV1_FEATURE_HEIGHT][CONV1_FEATURE_WIDTH],
		float output_features [CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH]
) {
	data_t in_buf[CONV1_FEATURES][CONV1_FEATURE_HEIGHT][CONV1_FEATURE_WIDTH];
	data_t w_buf[CONV1_KERNELS][CONV1_FEATURES][CONV1_KERNEL_HEIGHT][CONV1_KERNEL_WIDTH];
	acc_t  b_buf[CONV1_KERNELS];
#pragma HLS ARRAY_PARTITION variable=in_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_buf  complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_buf  complete dim=3
#pragma HLS ARRAY_PARTITION variable=w_buf  complete dim=4

	load_w1: for (int k = 0; k < CONV1_KERNELS; k++)
		for (int f = 0; f < CONV1_FEATURES; f++)
			for (int kh = 0; kh < CONV1_KERNEL_HEIGHT; kh++)
				for (int kw = 0; kw < CONV1_KERNEL_WIDTH; kw++)
					w_buf[k][f][kh][kw] = (data_t)input_kernels[k][f][kh][kw];

	load_b1: for (int k = 0; k < CONV1_KERNELS; k++)
		b_buf[k] = (acc_t)bias[k];

	load_in1: for (int f = 0; f < CONV1_FEATURES; f++)
		for (int h = 0; h < CONV1_FEATURE_HEIGHT; h++)
			for (int w = 0; w < CONV1_FEATURE_WIDTH; w++)
				in_buf[f][h][w] = (data_t)input_features[f][h][w];

	conv1_K: for (int k = 0; k < CONV1_KERNELS; k++) {
		conv1_FH: for (int fh = 0; fh < CONV1_CONVOLVED_FEATURE_HEIGHT; fh++) {
			conv1_FW: for (int fw = 0; fw < CONV1_CONVOLVED_FEATURE_WIDTH; fw++) {
#pragma HLS PIPELINE II=1

				acc_t products[32];
#pragma HLS ARRAY_PARTITION variable=products complete
				for (int f = 0; f < CONV1_FEATURES; f++)
					for (int kh = 0; kh < CONV1_KERNEL_HEIGHT; kh++)
						for (int kw = 0; kw < CONV1_KERNEL_WIDTH; kw++) {
#pragma HLS UNROLL
							products[f*CONV1_KERNEL_HEIGHT*CONV1_KERNEL_WIDTH + kh*CONV1_KERNEL_WIDTH + kw] =
								(acc_t)in_buf[f][fh+kh][fw+kw] * (acc_t)w_buf[k][f][kh][kw];
						}
				for (int i = CONV1_TAPS; i < 32; i++) {
#pragma HLS UNROLL
					products[i] = (acc_t)0;
				}

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
