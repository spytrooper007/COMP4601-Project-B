/*
 * C-simulation testbench for the conv5x5 HLS kernel.
 *
 * Generates a deterministic input image, weight set and bias, computes a
 * golden reference in plain double-precision C, runs the HLS kernel, and
 * reports the RMSE.  Self-contained: no external golden file needed
 * (same PASS/FAIL convention as the dft256 dft_test.cpp).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "conv5x5.h"

/* RMSE accumulator (from dft256/dft_test.cpp). */
struct Rmse {
    int num_sq; double sum_sq; double error;
    Rmse() { num_sq = 0; sum_sq = 0; error = 0; }
    double add_value(double d) { num_sq++; sum_sq += d * d; error = sqrt(sum_sq / num_sq); return error; }
};

static float input[IN_SIZE];
static float weights[W_SIZE];
static float bias[B_SIZE];
static float hw_out[OUT_SIZE];
static float gold[OUT_SIZE];

/* Plain-C golden reference (the software baseline for correctness). */
static void conv5x5_golden() {
    for (int oc = 0; oc < OUT_CH; oc++) {
        for (int oh = 0; oh < OUT_H; oh++) {
            for (int ow = 0; ow < OUT_W; ow++) {
                double acc = 0.0;
                for (int ic = 0; ic < IN_CH; ic++)
                    for (int kh = 0; kh < K; kh++)
                        for (int kw = 0; kw < K; kw++)
                            acc += (double)input[(ic * IN_H + (oh + kh)) * IN_W + (ow + kw)] *
                                   (double)weights[((oc * IN_CH + ic) * K + kh) * K + kw];
                double res = acc + (double)bias[oc];
                gold[(oc * OUT_H + oh) * OUT_W + ow] = (float)(res > 0.0 ? res : 0.0);
            }
        }
    }
}

int main() {
    /* Deterministic, reproducible test data in a sensible range. */
    for (int i = 0; i < IN_SIZE; i++) input[i]   = (float)((i % 17) - 8) * 0.1f;
    for (int i = 0; i < W_SIZE;  i++) weights[i] = (float)((i % 7)  - 3) * 0.05f;
    for (int i = 0; i < B_SIZE;  i++) bias[i]    = (float)(i - 3)  * 0.02f;

    conv5x5_golden();
    conv5x5(input, weights, bias, hw_out);

    Rmse rmse;
    for (int i = 0; i < OUT_SIZE; i++)
        rmse.add_value((double)hw_out[i] - (double)gold[i]);

    printf("----------------------------------------------\n");
    printf("   Conv5x5  RMSE\n");
    printf("%0.15f\n", rmse.error);
    printf("----------------------------------------------\n");

    if (rmse.error > 1e-3) {
        printf("*******************************************\n");
        printf("FAIL: Output DOES NOT match the golden output\n");
        printf("*******************************************\n");
        return 1;
    }
    printf("*******************************************\n");
    printf("PASS: The output matches the golden output!\n");
    printf("*******************************************\n");
    return 0;
}
