#ifndef CONV5X5_H
#define CONV5X5_H

#define IN_CH   1
#define IN_H    32
#define IN_W    32
#define K       5
#define OUT_CH  6
#define OUT_H   (IN_H - K + 1)
#define OUT_W   (IN_W - K + 1)

#define IN_SIZE   (IN_CH * IN_H * IN_W)
#define W_SIZE    (OUT_CH * IN_CH * K * K)
#define B_SIZE    (OUT_CH)
#define OUT_SIZE  (OUT_CH * OUT_H * OUT_W)

#ifdef __SYNTHESIS__
#include <ap_fixed.h>
typedef ap_fixed<20, 8>  data_t;
typedef ap_fixed<40, 16> acc_t;
#else
typedef float data_t;
typedef float acc_t;
#endif

void conv5x5(const float *input,
             const float *weights,
             const float *bias,
             float       *output);
#endif
