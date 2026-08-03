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
 
#include "test_padding_2.h"


bool test_padding_2(
		const int num_image
){
	float padded_features[CONV1_KERNELS][PADDED_FEATURES_HEIGHT][PADDED_FEATURES_WIDTH];
	padding_2(res_maxPool_1[num_image], padded_features);
	return (are_similar_3D(CONV1_KERNELS, PADDED_FEATURES_HEIGHT, PADDED_FEATURES_WIDTH, &padded_features[0][0][0], &res_addPadding_2[num_image][0][0][0], 0.000001));
	
}
