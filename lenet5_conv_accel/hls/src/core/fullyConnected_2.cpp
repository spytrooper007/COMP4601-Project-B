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

#include "fullyConnected_2.h"

// Applies a linear transformation
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
void fullyConnected_2(
		const float input_features [FC1_FEATURES],
		const float input_weights [FC2_FEATURES][FC1_FEATURES], 
		const float bias[FC2_FEATURES],         
		float output_features[FC2_FEATURES]
){

	float accumulated;

	uint16_t f;
	uint16_t nif;

	// Go through all the values of that feature
	fc2_F: for (f = 0; f < FC2_FEATURES; f++) {

		accumulated = 0.0f;

		// For each feature
		fc2_NIF: for (nif = 0; nif < FC1_FEATURES; nif++) {

			accumulated += input_features[nif] * input_weights[f][nif];
		}

		output_features[f] = accumulated + bias[f];
	}
}

