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

#ifndef CNN_SOFTMAX
#define CNN_SOFTMAX

#include <math.h>

#include "cnn_weights_and_bias_all_layers.h"
#include "stdint.h"

/*
	Results are transformed into a percentage,
	which states the probability of the image to be classified as that feature
*/
void softMax(
		const float input_features[FC2_FEATURES], 
		float output_features[FC2_FEATURES]);

#endif
