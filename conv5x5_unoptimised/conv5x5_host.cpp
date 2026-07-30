/*
 * XRT host program for the conv5x5 accelerator (runs on the KV260 ARM PS).
 *
 * Mirrors dft256/dft_host/dft_host.cpp:
 *   - loads the .xclbin, moves operands to the device, runs the kernel
 *   - checks the FPGA result against the same plain-C golden as the testbench
 *   - benchmarks average FPGA latency vs an ARM-only software convolution
 *
 * Build against XRT.  Run:
 *   ./conv5x5_host -x conv5x5.xclbin
 */
#include "cmdlineparser.h"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "conv5x5.h"

#define NRUNS 1000   /* repeat for a stable average */

static float g_in[IN_SIZE], g_w[W_SIZE], g_b[B_SIZE], g_gold[OUT_SIZE];

/* Plain-C reference convolution (also used as the ARM software baseline). */
static void conv5x5_sw(const float *in, const float *w, const float *b, float *out) {
    for (int oc = 0; oc < OUT_CH; oc++)
        for (int oh = 0; oh < OUT_H; oh++)
            for (int ow = 0; ow < OUT_W; ow++) {
                float acc = 0.0f;
                for (int ic = 0; ic < IN_CH; ic++)
                    for (int kh = 0; kh < K; kh++)
                        for (int kw = 0; kw < K; kw++)
                            acc += in[(ic * IN_H + (oh + kh)) * IN_W + (ow + kw)] *
                                   w[((oc * IN_CH + ic) * K + kh) * K + kw];
                float res = acc + b[oc];
                out[(oc * OUT_H + oh) * OUT_W + ow] = (res > 0.0f) ? res : 0.0f;
            }
}

int main(int argc, char **argv) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id",   "-d", "device index", "0");
    parser.parse(argc, argv);

    std::string binaryFile = parser.value("xclbin_file");
    int device_index       = stoi(parser.value("device_id"));
    if (argc < 3) { parser.printHelp(); return EXIT_FAILURE; }

    /* Deterministic test data (matches conv5x5_test.cpp). */
    for (int i = 0; i < IN_SIZE; i++) g_in[i] = (float)((i % 17) - 8) * 0.1f;
    for (int i = 0; i < W_SIZE;  i++) g_w[i]  = (float)((i % 7)  - 3) * 0.05f;
    for (int i = 0; i < B_SIZE;  i++) g_b[i]  = (float)(i - 3)  * 0.02f;
    conv5x5_sw(g_in, g_w, g_b, g_gold);

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    auto krnl = xrt::kernel(device, uuid, "conv5x5");

    auto bo_in  = xrt::bo(device, sizeof(float) * IN_SIZE,  krnl.group_id(0));
    auto bo_w   = xrt::bo(device, sizeof(float) * W_SIZE,   krnl.group_id(1));
    auto bo_b   = xrt::bo(device, sizeof(float) * B_SIZE,   krnl.group_id(2));
    auto bo_out = xrt::bo(device, sizeof(float) * OUT_SIZE, krnl.group_id(3));

    auto p_in  = bo_in.map<float *>();
    auto p_w   = bo_w.map<float *>();
    auto p_b   = bo_b.map<float *>();
    auto p_out = bo_out.map<float *>();

    for (int i = 0; i < IN_SIZE; i++) p_in[i] = g_in[i];
    for (int i = 0; i < W_SIZE;  i++) p_w[i]  = g_w[i];
    for (int i = 0; i < B_SIZE;  i++) p_b[i]  = g_b[i];
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = krnl(bo_in, bo_w, bo_b, bo_out);
    run.wait();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    /* Correctness vs golden. */
    double sum_sq = 0.0;
    for (int i = 0; i < OUT_SIZE; i++) {
        double d = (double)p_out[i] - (double)g_gold[i];
        sum_sq += d * d;
    }
    double rmse = std::sqrt(sum_sq / OUT_SIZE);
    std::cout << "----------------------------------------------\n";
    printf("Conv5x5 RMSE : %0.15f\n", rmse);
    std::cout << (rmse > 1e-3 ? "FAIL: does not match golden output\n"
                              : "PASS: matches golden output\n");
    std::cout << "----------------------------------------------\n";

    /* FPGA timing. */
    double fpga_total = 0.0;
    for (int r = 0; r < NRUNS; r++) {
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto run2 = krnl(bo_in, bo_w, bo_b, bo_out);
        run2.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        fpga_total += std::chrono::duration<double>(t1 - t0).count();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
    double fpga_avg = fpga_total / NRUNS;

    /* ARM software timing. */
    static float sw_out[OUT_SIZE];
    auto s0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NRUNS; r++) conv5x5_sw(g_in, g_w, g_b, sw_out);
    auto s1 = std::chrono::high_resolution_clock::now();
    double sw_avg = std::chrono::duration<double>(s1 - s0).count() / NRUNS;

    std::cout << "\n--- Performance (avg over " << NRUNS << " runs) ---\n";
    printf("FPGA time : %0.3f us\n", fpga_avg * 1e6);
    printf("ARM  time : %0.3f us\n", sw_avg  * 1e6);
    printf("Speedup   : %0.2fx\n",   sw_avg / fpga_avg);
    return 0;
}
