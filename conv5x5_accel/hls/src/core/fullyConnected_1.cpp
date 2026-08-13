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

#include "fullyConnected_1.h"

// Applies a linear transformation
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
void fullyConnected_1(
		const float input_features [FC1_INPUT_FEATURES],
		const float input_weights [FC1_FEATURES][FC1_INPUT_FEATURES],
		const float bias[FC1_FEATURES],
		float output_features[FC1_FEATURES]
) {

	float accumulated;

	uint16_t f;
	uint16_t nif;

	// Go through all the values of that feature
	fc1_F: for (f = 0; f < FC1_FEATURES; f++) {

#pragma HLS PIPELINE II=1

		accumulated = 0.0f;

		// For each feature
		fc1_NIF: for (nif = 0; nif < FC1_INPUT_FEATURES; nif++) {

#pragma HLS EXPRESSION_BALANCE

			accumulated += input_features[nif] * input_weights[f][nif];
		}

		output_features[f] = accumulated + bias[f];
	}
}
