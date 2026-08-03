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
 
#ifndef TEST_MATRIX_COMPARISON
#define TEST_MATRIX_COMPARISON

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "test.h"

bool are_equal_3D(
	const uint16_t MAXD1,
	const uint16_t MAXD2,
	const uint16_t MAXD3,
	float *current,
	const float *expected
);

bool are_similar_3D(
	const uint16_t MAXD1,
	const uint16_t MAXD2,
	const uint16_t MAXD3,
	float *current,
	const float *expected,
	const float tolerance
);


bool are_similar_1D(
	const uint16_t MAXD1,
	float *current,
    const float *expected,
	const float tolerance
);

#endif