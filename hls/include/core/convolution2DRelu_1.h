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

#ifndef CNN_CONVRELU_1
#define CNN_CONVRELU_1

#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"

// Applies a 2D Convolution according to: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
void convolution2DRelu_1(
		const float input_kernels [CONV1_KERNELS][CONV1_FEATURES][CONV1_KERNEL_HEIGHT][CONV1_KERNEL_WIDTH],  
		const float bias[CONV1_KERNELS],           
		const float input_features [CONV1_FEATURES][CONV1_FEATURE_HEIGHT][CONV1_FEATURE_WIDTH], 
		float output_features [CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH]      
		);

#endif
