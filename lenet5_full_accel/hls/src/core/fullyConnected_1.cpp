/*
 * Copyright (c) 2021-2023 Universitat Politècnica de València
 * Distributed under MIT License (See accompanying file LICENSE.txt)
 *
 * ACCELERATED FC1: fixed-point, 8 parallel MACs/cycle (strided pipelined inner
 * loop + cyclic-partitioned operands).
 */
#include "fullyConnected_1.h"

#ifdef __SYNTHESIS__
#include <ap_fixed.h>
typedef ap_fixed<20, 8>  fdata_t;
typedef ap_fixed<40, 16> facc_t;
#else
typedef float fdata_t;
typedef float facc_t;
#endif

#define FC1_UF 16

void fullyConnected_1(
		const float input_features [FC1_INPUT_FEATURES],
		const float input_weights [FC1_FEATURES][FC1_INPUT_FEATURES],
		const float bias[FC1_FEATURES],
		float output_features[FC1_FEATURES]
) {
#pragma HLS ARRAY_PARTITION variable=input_features cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=input_weights  cyclic factor=16 dim=2

	fc1_F: for (int f = 0; f < FC1_FEATURES; f++) {
		facc_t acc = 0;
		fc1_NIF: for (int base = 0; base < FC1_INPUT_FEATURES; base += FC1_UF) {
#pragma HLS PIPELINE II=1
			facc_t lane[FC1_UF];
#pragma HLS ARRAY_PARTITION variable=lane complete
			for (int j = 0; j < FC1_UF; j++) {
#pragma HLS UNROLL
				int idx = base + j;
				lane[j] = (idx < FC1_INPUT_FEATURES)
					? (facc_t)((fdata_t)input_features[idx] * (fdata_t)input_weights[f][idx])
					: (facc_t)0;
			}
			facc_t s = 0;
			for (int j = 0; j < FC1_UF; j++) {
#pragma HLS UNROLL
				s += lane[j];
			}
			acc += s;
		}
		output_features[f] = (float)(acc + (facc_t)bias[f]);
	}
}
