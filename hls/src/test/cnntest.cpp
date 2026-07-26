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

#define _CRT_SECURE_NO_WARNINGS

#include "test.h"
#include <string>

void test_function(const char* name, bool (*f)(const int), int image_number, int last_image) {
	bool match;

	printf("*\t - %s :\n", name);
	if (last_image == -1) {
		match = (*f)(image_number);
		printf("The results of the %s operation on image %d are %s\n", name, image_number, match ? "correct" : "incorrect");
	}
	else {
		int sucess = 0;
		for (int i = image_number; i < last_image; i++) {
			match = (*f)(i);
			if (match) sucess++;
			printf("The results of the %s operation on image %d are %s\n\n", name, i, match ? "correct" : "incorrect");
		}
		printf("The accuracy of the %s operation is %f\n", name, ((float)sucess * 100) / (last_image - image_number));
	}
}

int main(int argc, char* argv[]) {

	int operation = 0; // Default to test_cnn
	int image_number = 0; // Default to first image
	int last_image = -1; // Default to no images range
	int name_index = 0; // Default to test_cnn
	bool (*f)(const int);
	bool (*ff)(const int, const float input_image[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH]);
	bool match;

	f = nullptr;
	ff = nullptr;

	const char* name[10] = {
			"test_cnn",
			"test_padding_1",
			"test_convolution2DRelu_1",
			"test_maxPool_1",
			"test_padding_2",
			"test_convolution2DRelu_2",
			"test_maxPool_2",
			"test_flattenLayer",
			"test_fullyConnected_1",
			"test_fullyConnected_2"
	};

	if (argc > 1) {
		operation = std::stoi(argv[1]);
		name_index = operation;
	}
	if (argc > 2) {
		image_number = std::stoi(argv[2]);
	}
	if (argc > 3) {
		last_image = std::stoi(argv[3]);
	}

	printf("*******************************\n");
	printf("* LENET-5 CNN testing program *\n");
	printf("*******************************\n");

	switch (operation) {

	case 0: {
		f = test_cnn;
	}
		  break;

	case 1: {
		f = test_padding_1;
	}
		  break;

	case 2: {
		f = test_convolution2DRelu_1;
	}
		  break;

	case 3: {
		f = test_maxPool_1;
	}
		  break;

	case 4: {
		f = test_padding_2;
	}
		  break;

	case 5: {
		f = test_convolution2DRelu_2;
	}
		  break;

	case 6: {
		f = test_maxPool_2;
	}
		   break;

	case 7: {
		f = test_flattenLayer;
	}
		   break;

	case 8: {
		f = test_fullyConnected_1;
	}
		   break;

	case 9: {
		f = test_fullyConnected_2;
	}
		   break;

	}

	if (operation < 10) {
		test_function(name[name_index], f, image_number, last_image);
	}

	printf("Done");

	return 0;
}
