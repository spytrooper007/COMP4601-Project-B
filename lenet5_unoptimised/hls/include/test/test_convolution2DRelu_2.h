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

#ifndef TEST_CONVOLUTION2DRELU_2
#define TEST_CONVOLUTION2DRELU_2

#include "test.h"
#include "matrix_comparison.h"
#include "oracle/cnn_ConvRelu2_res.h"
#include "../core/convolution2DRelu_2.h"


bool test_convolution2DRelu_2(
		const int num_image
);

#endif
