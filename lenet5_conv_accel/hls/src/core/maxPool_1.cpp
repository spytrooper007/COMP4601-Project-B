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

#include "maxPool_1.h"

// Gets the maximum values in a squared region delimited by "step"
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
void maxPool_1(
	const float input_features [CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH], 
	float output_features [CONV1_KERNELS][CONV1_MAXPOOL_FINAL_HEIGHT][CONV1_MAXPOOL_FINAL_WIDTH]
) {

	float maxValue;

	uint16_t f;
	uint16_t hm;
	uint16_t wm;
	uint16_t hf;
	uint16_t wf;

	// For each final feature
	max1_F: for (f = 0; f < CONV1_KERNELS; f++) {

		// Go through final CONV1_KERNELS rows and columns
		max1_HM: for (hm = 0; hm < CONV1_MAXPOOL_FINAL_HEIGHT; hm++) {

			max1_WM: for (wm = 0; wm < CONV1_MAXPOOL_FINAL_WIDTH; wm++) {

				maxValue = 0.0f;
				// Go through final CONV1_KERNELS rows and columns according to the selected step
				max1_HF: for (hf = 0; hf < CONV1_MAXPOOL_STEP; hf++) {

					max1_WF: for (wf = 0; wf < CONV1_MAXPOOL_STEP; wf++) {

						// Determine the maximum value within this region
						if (input_features[f][hm * CONV1_MAXPOOL_STEP + hf][wm* CONV1_MAXPOOL_STEP + wf] > maxValue) {
							maxValue = input_features[f][hm * CONV1_MAXPOOL_STEP + hf][wm * CONV1_MAXPOOL_STEP + wf];
						}
					}
				}

				// Assign the maximum value found to the final feature
				output_features[f][hm][wm] = maxValue;
			}
		}

	}
}
