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
 
#include "test_cnn.h"


bool test_cnn(
		const int num_image
) {
	// Classify the input image according to the defined CNN
	float classification[FC2_FEATURES];
	cnn(INPUT_IMAGES[num_image],classification);
	float max = classification[0];
	uint8_t index = 0;
	for (uint8_t i = 1; i < FC2_FEATURES; i++) {
		if (classification[i] > max) {
			max = classification[i];
			index = i;
		}
	}
	return (index == res_cnn_prediction[num_image]);
}

