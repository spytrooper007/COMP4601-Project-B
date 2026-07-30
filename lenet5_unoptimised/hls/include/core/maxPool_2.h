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

#ifndef CNN_MAX_2
#define CNN_MAX_2

#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"

// Gets the maximum values in a sqaured region delimited by "step"
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
void maxPool_2(
	const float input_features [CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH], 
	float output_features [CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT]
);

#endif
