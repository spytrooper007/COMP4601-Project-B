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

#ifndef CNN_FLATTEN_LAYER
#define CNN_FLATTEN_LAYER

#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"

/*
 * https://pytorch.org/docs/stable/generated/torch.flatten.html
 */
void flattenLayer(
		const float input_features [CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT], 
		float output_features[FC1_INPUT_FEATURES]);

#endif
