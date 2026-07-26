/*
 * Copyright (c) 2021-2023 Universitat Politècnica de València
 * Authors: David de Andrés and Juan Carlos Ruiz
 *          Fault-Tolerant Systems
 *          Instituto ITACA
 *          Universitat Politècnica de València
 *
 * Distributed under MIT License
 * (See accompanying file LICENSE.txt)
 */

#include "convolution2DRelu_2.h"

// Applies a 2D Convolution according to: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
void convolution2DRelu_2(
		const float input_kernels [CONV2_KERNELS][CONV1_KERNELS][CONV2_KERNEL_HEIGHT][CONV2_KERNEL_WIDTH],  
		const float bias[CONV2_KERNELS],           
		const float input_features [CONV1_KERNELS][PADDED_FEATURES_WIDTH][PADDED_FEATURES_HEIGHT], 
		float output_features [CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH]      
) {

	float accumulated;

	uint16_t k;
	uint16_t fh;
	uint16_t fw;
	uint16_t kh;
	uint16_t kw;
	uint16_t f;

	// For each kernel
	conv2_K: for (k = 0; k < CONV2_KERNELS; k++) {

		// Go through features rows and columns
		conv2_FH: for (fh = 0; fh < CONV2_CONVOLVED_FEATURE_HEIGHT; fh++) {
			conv2_FW: for (fw = 0; fw < CONV2_CONVOLVED_FEATURE_WIDTH; fw++) {

#pragma HLS PIPELINE II=1

				// Reset accumulated value
				accumulated = 0.0f;
				// Go through the kernel rows and columns
				conv2_KH: for (kh = 0; kh < CONV2_KERNEL_HEIGHT; kh++) {
					conv2_KW: for (kw = 0; kw < CONV2_KERNEL_WIDTH; kw++) {

						// Convolve each feature with the corresponding kernel and add the result
						conv2_F: for (f = 0; f < CONV1_KERNELS; f++) {

#pragma HLS EXPRESSION_BALANCE

							accumulated += input_features[f][fh + kh][fw + kw] * input_kernels[k][f][kh][kw];
						}

					}

				}
				// Add bias and assign result
				accumulated += bias[k];
				output_features[k][fh][fw] = (accumulated > 0.0f) ? accumulated : 0.0f;

			}

		}

	}
}
