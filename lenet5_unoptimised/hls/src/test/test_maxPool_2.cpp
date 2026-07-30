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
 
#include "test_maxPool_2.h"

bool test_maxPool_2(
		const int num_image
){
	// After the MaxPooling layer the 3 features are reduced to size 14x14
	float output[CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT];
	maxPool_2(res_conv2DRelu_2[num_image], output);
	return (are_similar_3D(CONV2_KERNELS, CONV2_MAXPOOL_FINAL_HEIGHT, CONV2_MAXPOOL_FINAL_WIDTH,&output[0][0][0],&res_maxPool_2[num_image][0][0][0],0.000001));
}
