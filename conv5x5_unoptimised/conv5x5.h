#ifndef CONV5X5_H
#define CONV5X5_H

/*
 * Single 5x5 2D convolution layer + ReLU, sized to match LeNet-5's first
 * convolution (Conv1): a 32x32 single-channel (already zero-padded 28+2*2)
 * input is convolved with OUT_CH 5x5 kernels to produce OUT_CH 28x28 maps.
 *
 * This is the standalone hardware/software accelerator used to validate the
 * HLS -> bitstream -> PS/PL flow before extending to the full network in
 * ../lenet5.  Structured the same way as the dft256 lab (kernel + self-checking
 * C-sim testbench + XRT host that benchmarks FPGA vs ARM).
 *
 * Data type is float for now (matches the ../lenet5 reference and the "start in
 * floating point" plan); fixed-point (ap_fixed) is a later optimisation.
 */

#define IN_CH   1                       /* input channels                     */
#define IN_H    32                      /* input height  (28 + 2*2 padding)   */
#define IN_W    32                      /* input width                        */
#define K       5                       /* kernel size (5x5)                  */
#define OUT_CH  6                        /* number of output feature maps      */
#define OUT_H   (IN_H - K + 1)           /* 28                                 */
#define OUT_W   (IN_W - K + 1)           /* 28                                 */

#define IN_SIZE   (IN_CH * IN_H * IN_W)              /* 1024 */
#define W_SIZE    (OUT_CH * IN_CH * K * K)           /* 150  */
#define B_SIZE    (OUT_CH)                           /* 6    */
#define OUT_SIZE  (OUT_CH * OUT_H * OUT_W)           /* 4704 */

/* Flat-buffer interface so the XRT host can map each argument to an xrt::bo. */
void conv5x5(const float *input,    /* [IN_CH][IN_H][IN_W]        */
             const float *weights,  /* [OUT_CH][IN_CH][K][K]      */
             const float *bias,     /* [OUT_CH]                   */
             float       *output);  /* [OUT_CH][OUT_H][OUT_W]     */

#endif
