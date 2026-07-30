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

#ifndef CNN_FC_1
#define CNN_FC_1

#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"

void fullyConnected_1(
		const float input_features [FC1_INPUT_FEATURES],
		const float input_weights [FC1_FEATURES][FC1_INPUT_FEATURES], 
		const float bias[FC1_FEATURES],         
		float output_features[FC1_FEATURES]
);

#endif
