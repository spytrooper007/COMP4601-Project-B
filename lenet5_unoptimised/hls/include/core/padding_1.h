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

#ifndef CNN_PADDING_1
#define CNN_PADDING_1

#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"

void padding_1(
		const float features[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH], 
		float padded_features[IMAGE_FEATURES][PADDED_IMAGE_HEIGHT][PADDED_IMAGE_WIDTH]);

#endif
