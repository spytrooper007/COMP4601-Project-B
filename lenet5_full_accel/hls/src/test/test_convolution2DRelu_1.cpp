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

#include "test_convolution2DRelu_1.h"

bool test_convolution2DRelu_1(
		const int num_image
){

	float output_features [CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH];
	convolution2DRelu_1(KERNEL_CONV_1, BIAS_CONV_1, res_addPadding_1[num_image], output_features);
	return (are_similar_3D(CONV1_KERNELS, CONV1_CONVOLVED_FEATURE_HEIGHT, CONV1_CONVOLVED_FEATURE_WIDTH, &output_features[0][0][0], &res_conv2DRelu_1[num_image][0][0][0], 0.0001));
}
