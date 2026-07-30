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

#include "test_flattenLayer.h"

bool test_flattenLayer(
		const int num_image
){

	// After the MaxPooling layer the 3 features are reduced to size 14x14
	float output[FC1_INPUT_FEATURES];
	flattenLayer(res_maxPool_2[num_image], output);
	return (are_similar_1D(FC1_INPUT_FEATURES,&output[0],&res_flattenLayer[num_image][0],0.000001));

}
