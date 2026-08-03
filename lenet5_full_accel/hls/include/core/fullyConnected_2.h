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

#ifndef CNN_FC_2
#define CNN_FC_2

#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"

void fullyConnected_2(
		const float input_features [FC1_FEATURES],
		const float input_weights [FC2_FEATURES][FC1_FEATURES],
		const float bias[FC2_FEATURES],         
		float output_features[FC2_FEATURES]
);

#endif
