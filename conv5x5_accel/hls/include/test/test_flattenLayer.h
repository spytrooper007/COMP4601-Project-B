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
 
#ifndef TEST_FLATTENLAYER
#define TEST_FLATTENLAYER

#include "test.h"
#include "oracle/cnn_FlattenLayer_res.h"
#include "../core/flattenLayer.h"

/*
 * https://pytorch.org/docs/stable/generated/torch.flatten.html
 */
bool test_flattenLayer(
		const int num_image
);

#endif
