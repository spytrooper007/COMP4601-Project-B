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
 
#include "test_convolution2DRelu_2.h"

bool test_convolution2DRelu_2(
		const int num_image
){

	float output_features [CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH];
	convolution2DRelu_2(KERNEL_CONV_2, BIAS_CONV_2, res_addPadding_2[num_image], output_features);
	return (are_similar_3D(CONV2_KERNELS, CONV2_CONVOLVED_FEATURE_HEIGHT, CONV2_CONVOLVED_FEATURE_WIDTH, &output_features[0][0][0], &res_conv2DRelu_2[num_image][0][0][0], 0.0001));
	
}
