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

#include "flattenLayer.h"

/*
 * https://pytorch.org/docs/stable/generated/torch.flatten.html
 */
void flattenLayer(
		const float input_features [CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT], 
		float output_features[FC1_INPUT_FEATURES]
) {

	   uint16_t f;
	   uint16_t w;
	   uint16_t h;

	   // For each feature
	flat_F: for (f = 0; f < CONV2_KERNELS; f++) {

		// Go through rows and columns
		flat_H: for (h = 0; h < CONV2_MAXPOOL_FINAL_HEIGHT; h++){
			flat_W: for (w = 0; w < CONV2_MAXPOOL_FINAL_WIDTH; w++){

				// Copy each value into a unidimensional array
				output_features[f*CONV2_MAXPOOL_FINAL_HEIGHT*CONV2_MAXPOOL_FINAL_WIDTH+ h*CONV2_MAXPOOL_FINAL_WIDTH+ w] = input_features[f][h][w];
			}
		}

	}

}
