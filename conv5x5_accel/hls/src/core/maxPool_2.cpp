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

#include "maxPool_2.h"

// Gets the maximum values in a squared region delimited by "step"
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
void maxPool_2(
	const float input_features [CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH], 
	float output_features [CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT]
){

#pragma HLS ARRAY_PARTITION variable=input_features type=cyclic factor=2 dim=2

	float maxValue;

	uint16_t f;
	uint16_t hm;
	uint16_t wm;
	uint16_t hf;
	uint16_t wf;

	// For each final feature
	max2_F: for (f = 0; f < CONV2_KERNELS; f++) {

		// Go through final CONV1_KERNELS rows and columns
		max2_HM: for (hm = 0; hm < CONV2_MAXPOOL_FINAL_HEIGHT; hm++) {

			max2_WM: for (wm = 0; wm < CONV2_MAXPOOL_FINAL_WIDTH; wm++) {

#pragma HLS PIPELINE II=1

				maxValue = 0.0f;
				// Go through final CONV1_KERNELS rows and columns according to the selected step
				max2_HF: for (hf = 0; hf < CONV2_MAXPOOL_STEP; hf++) {

					max2_WF: for (wf = 0; wf < CONV2_MAXPOOL_STEP; wf++) {

						// Determine the maximum value within this region
						if (input_features[f][hm * CONV2_MAXPOOL_STEP + hf][wm* CONV2_MAXPOOL_STEP + wf] > maxValue) {
							maxValue = input_features[f][hm * CONV2_MAXPOOL_STEP + hf][wm * CONV2_MAXPOOL_STEP + wf];
						}
					}
				}

				// Assign the maximum value found to the final feature
				output_features[f][hm][wm] = maxValue;
			}
		}

	}
}
