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
 
#include "cnn.h"

// Implements a Convolutional Neural Network consisting in 2 convolution layers and 2 fully connected layers
void cnn(
	const float input_image[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH],
	float classification[FC2_FEATURES]
) {

	#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
	#pragma HLS INTERFACE mode=m_axi port=input_image depth=1*28*28 bundle=gmem
	#pragma HLS INTERFACE mode=m_axi port=classification depth=10 bundle=gmem

	float padded_image[IMAGE_FEATURES][PADDED_IMAGE_HEIGHT][PADDED_IMAGE_WIDTH];
	float first_convolution[CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH];
	float first_convolution_relu[CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH];
	float first_convolution_max[CONV1_KERNELS][CONV1_MAXPOOL_FINAL_HEIGHT][CONV1_MAXPOOL_FINAL_WIDTH];

	float first_convolution_padded[CONV1_KERNELS][PADDED_FEATURES_WIDTH][PADDED_FEATURES_HEIGHT];
	float second_convolution[CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH];
	float second_convolution_relu[CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH];
	float second_convolution_max[CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT];
	
	float flatten[FC1_INPUT_FEATURES];
	float fc1[FC1_FEATURES];

	padding_1(input_image, padded_image);
	convolution2DRelu_1(KERNEL_CONV_1, BIAS_CONV_1, padded_image, first_convolution_relu);
	maxPool_1(first_convolution_relu, first_convolution_max);
	
	padding_2(first_convolution_max, first_convolution_padded);
	convolution2DRelu_2(KERNEL_CONV_2, BIAS_CONV_2, first_convolution_padded, second_convolution_relu);
	maxPool_2(second_convolution_relu, second_convolution_max);
	
	flattenLayer(second_convolution_max, flatten);
	fullyConnected_1(flatten, WEIGHTS_FC1, BIAS_FC1, fc1);
	fullyConnected_2(fc1, WEIGHTS_FC2, BIAS_FC2, classification);
}