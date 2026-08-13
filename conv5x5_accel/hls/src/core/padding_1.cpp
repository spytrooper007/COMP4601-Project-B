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

#include "padding_1.h"

// Add 0.0 as padding on the top and bottom rows and left and right columns of provided features
void padding_1(
		const float features[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH], 
		float padded_features[IMAGE_FEATURES][PADDED_IMAGE_HEIGHT][PADDED_IMAGE_WIDTH]
) {

	uint16_t f;
	uint16_t h;
	uint16_t w;

	// For each feature
	pad1_F: for (f = 0; f < IMAGE_FEATURES; f++) {

		// Go through rows and columns
		pad1_H: for (h = 0; h < IMAGE_HEIGHT + 2*PADDING ; h++) {

			pad1_W: for (w = 0; w < IMAGE_WIDTH + 2*PADDING; w++) {

#pragma HLS PIPELINE II=1

				// Fill left and right columns with 0.0
				// Fill top and bottom rows with 0.0
				if (w < PADDING || w > IMAGE_WIDTH + PADDING -1 ||
					h < PADDING || h > IMAGE_HEIGHT + PADDING -1) {
					padded_features[f][h][w] = 0.0f;

				}
				// Fill the rest of the image with the actual pixel
				else {
					padded_features[f][h][w]=features[f][h-PADDING][w-PADDING];
				}

			}
		}

	}
}

