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

#include "convolution2DRelu_1.h"

// Applies a 2D Convolution according to: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
void convolution2DRelu_1(
		const float input_kernels [CONV1_KERNELS][CONV1_FEATURES][CONV1_KERNEL_HEIGHT][CONV1_KERNEL_WIDTH],  
		const float bias[CONV1_KERNELS],           
		const float input_features [CONV1_FEATURES][CONV1_FEATURE_HEIGHT][CONV1_FEATURE_WIDTH], 
		float output_features [CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH]      
) {

	float accumulated;

	uint16_t k;
	uint16_t fh;
	uint16_t fw;
	uint16_t kh;
	uint16_t kw;
	uint16_t f;

	// For each kernel
	conv1_K: for (k = 0; k < CONV1_KERNELS; k++) {

		// Go through features rows and columns
		conv1_FH: for (fh = 0; fh < CONV1_CONVOLVED_FEATURE_HEIGHT; fh++) {
			conv1_FW: for (fw = 0; fw < CONV1_CONVOLVED_FEATURE_WIDTH; fw++) {

				// Reset accumulated value
				accumulated = 0.0f;
				// Go through the kernel rows and columns
				conv1_KH: for (kh = 0; kh < CONV1_KERNEL_HEIGHT; kh++) {
					conv1_KW: for (kw = 0; kw < CONV1_KERNEL_WIDTH; kw++) {

						// Convolve each feature with the corresponding kernel and add the result
						conv1_F: for (f = 0; f < CONV1_FEATURES; f++) {

							// Multiply each value by its corresponding weight
							accumulated+= input_features[f][fh+kh][fw+kw]*input_kernels[k][f][kh][kw];
						}

					}

				}

				// Add bias and assign result
				float res = accumulated + bias[k];
				output_features[k][fh][fw] = (res > 0.0f) ? res : 0.0f;

			}

		}

	}
}
