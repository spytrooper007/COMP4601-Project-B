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

#ifndef CNN_PADDING_2
#define CNN_PADDING_2

#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"

void padding_2(
		const float features[CONV1_KERNELS][CONV1_MAXPOOL_FINAL_HEIGHT][CONV1_MAXPOOL_FINAL_WIDTH], 
		float padded_features[CONV1_KERNELS][PADDED_FEATURES_HEIGHT][PADDED_FEATURES_WIDTH]);

#endif
