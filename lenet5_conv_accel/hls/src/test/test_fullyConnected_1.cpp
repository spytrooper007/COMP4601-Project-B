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
 
#include "test_fullyConnected_1.h"

bool test_fullyConnected_1(
		const int num_image
){
	
	// After the MaxPooling layer the 3 features are reduced to size 14x14
	float output[FC1_FEATURES];
	fullyConnected_1(res_flattenLayer[num_image], WEIGHTS_FC1, BIAS_FC1, output);
	return (are_similar_1D(FC1_FEATURES,&output[0],&res_fullyConnected_1[num_image][0],0.0001));

}
