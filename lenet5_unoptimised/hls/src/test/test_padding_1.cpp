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
 
#include "test_padding_1.h"

bool test_padding_1(
	const int num_image
){
	float padded_image[IMAGE_FEATURES][PADDED_IMAGE_HEIGHT][PADDED_IMAGE_WIDTH];
	padding_1(INPUT_IMAGES[num_image], padded_image);
	return (are_equal_3D(IMAGE_FEATURES, IMAGE_HEIGHT, IMAGE_WIDTH, &padded_image[0][0][0], &res_addPadding_1[num_image][0][0][0]));
}
