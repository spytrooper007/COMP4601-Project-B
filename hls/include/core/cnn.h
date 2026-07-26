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

#ifndef CNN

#define CNN

#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"
#include "layers.h"
#include "classification.h"

void cnn(
	const float input_image[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH],
	float classification[FC2_FEATURES]
);

#endif
