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

#ifndef CNN_CLASSIFICATION
#define CNN_CLASSIFICATION

#include "cnn_defines.h"

#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "cnn_weights_and_bias_all_layers.h"

// Determine the classification provided by the CNN
int classification(const float input_features[FC2_FEATURES]);

#endif
