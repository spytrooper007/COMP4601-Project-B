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
 
#include "test_fullyConnected_2.h"

bool test_fullyConnected_2(
		const int num_image
		){
	
	// After FC2 output will contain 10 values
	float output[FC2_FEATURES];
	fullyConnected_2(res_fullyConnected_1[num_image], WEIGHTS_FC2, BIAS_FC2, output);
	return (are_similar_1D(FC2_FEATURES,&output[0],&res_fullyConnected_2[num_image][0],0.00001));
}
