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

#ifndef TEST_H

#define TEST_H

const bool debug = false;

#include <stdio.h>
#include <math.h>
#include <stdint.h>


#include "../core/cnn_defines.h"
#include "mnist_input_images.h"

#include "test_padding_1.h"
#include "test_convolution2DRelu_1.h"
#include "test_maxPool_1.h"

#include "test_padding_2.h"
#include "test_convolution2DRelu_2.h"
#include "test_maxPool_2.h"

#include "test_flattenLayer.h"
#include "test_fullyConnected_1.h"
#include "test_fullyConnected_2.h"
#include "test_classification.h"

#include "test_cnn.h"

#endif
