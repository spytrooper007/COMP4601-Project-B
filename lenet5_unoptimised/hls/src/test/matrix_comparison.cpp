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
 
#include "matrix_comparison.h"


bool are_similar_3D(
	const uint16_t MAXD1,
	const uint16_t MAXD2,
	const uint16_t MAXD3,
	float *current,
    const float *expected,
	const float tolerance
){


	int f = 0;
	int w = 0;
	int h = 0;
	int i = 0;
		

	for (f = 0; f < MAXD1; f++) {
		for (h = 0; h < MAXD2; h++) {
			for (w = 0; w < MAXD3; w++) {
				i = f*MAXD2*MAXD3 + h*MAXD3 + w;
				if (debug) printf("Expected [%d][%d][%d] = %.5f, Current [%d][%d][%d] = %.5f\n", f, h, w, expected[i], f, h, w, current[i]);
				if (fabs(current[i] - expected[i]) > tolerance) {
					return false;
				}
			}
		}
	}

	return true;
}

bool are_similar_1D(
	const uint16_t MAXD1,
	float *current,
    const float *expected,
	const float tolerance
){


	int f = 0;
	int w = 0;
	int h = 0;
	int i = 0;
		
	for (f = 0; f < MAXD1; f++) {
		if (fabs(current[f] - expected[f]) > tolerance) {
			if (debug) printf("Expected [%d] = %.5f, Current [%d] = %.5f\n", f, expected[f], f, current[f]);
			return false;
		}
	}

	return true;
}


bool are_equal_3D(
	const uint16_t MAXD1,
	const uint16_t MAXD2,
	const uint16_t MAXD3,
	float *current,
	const float *expected
){
	return are_similar_3D(MAXD1, MAXD2, MAXD3, current, expected, 0.0);
}


