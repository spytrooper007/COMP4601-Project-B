/*
 * XRT host program for the LeNet-5 CNN accelerator (runs on the KV260 ARM PS).
 *
 * The trained weights are compiled INTO the kernel (static const arrays in
 * cnn_weights_and_bias_all_layers.h), so the host passes ONLY the 28x28 image
 * and reads back the 10 output logits. Two modes:
 *
 *   1. MNIST accuracy   (default): classify N bundled MNIST test images,
 *                        compare argmax to the true label, report accuracy,
 *                        and benchmark FPGA vs ARM software.
 *   2. Custom image     (-i file): classify your own 28x28 image, print an
 *                        ASCII preview, per-digit soft-max probabilities, and
 *                        the predicted digit.
 *
 * Build: see UserConfig.cmake (links the core/*.cpp for the ARM baseline).
 * Run examples:
 *   ./lenet5_host -x lenet5.bin                 # accuracy over 100 images
 *   ./lenet5_host -x lenet5.bin -n 50           # first 50 images
 *   ./lenet5_host -x lenet5.bin -i mydigit.txt              # raw 0..255 input
 *   ./lenet5_host -x lenet5.bin -i mydigit.txt --format std # already normalized
 */
#include "cmdlineparser.h"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "cnn.h"                        // cnn() proto, defines, baked-in weights
#include "mnist_input_images.h"         // INPUT_IMAGES[NUM_IMAGES][1][28][28]
#include "mnist_labels_and_predictions.h" // res_cnn_real[NUM_IMAGES] (true labels)

/* Classic MNIST normalisation used to train the reference model:
 *   normalised = (pixel/255 - 0.1307) / 0.3081
 * (background pixel 0 -> -0.4242, which is what the bundled data contains). */
static const float MNIST_MEAN = 0.1307f;
static const float MNIST_STD  = 0.3081f;

static int argmax10(const float *v) {
    int idx = 0; float m = v[0];
    for (int i = 1; i < FC2_FEATURES; i++) if (v[i] > m) { m = v[i]; idx = i; }
    return idx;
}

static void softmax10(const float *in, float *out) {
    float mx = in[0];
    for (int i = 1; i < FC2_FEATURES; i++) if (in[i] > mx) mx = in[i];
    float sum = 0.0f;
    for (int i = 0; i < FC2_FEATURES; i++) { out[i] = std::exp(in[i] - mx); sum += out[i]; }
    for (int i = 0; i < FC2_FEATURES; i++) out[i] /= sum;
}

/* ASCII preview of a normalised image (de-normalise back to 0..1 intensity). */
static void print_ascii(const float img[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH]) {
    const char *ramp = " .:-=+*#%@";
    for (int h = 0; h < IMAGE_HEIGHT; h++) {
        for (int w = 0; w < IMAGE_WIDTH; w++) {
            float intensity = img[0][h][w] * MNIST_STD + MNIST_MEAN;   // -> ~0..1
            if (intensity < 0) intensity = 0; if (intensity > 1) intensity = 1;
            int level = (int)(intensity * 9.0f + 0.5f);
            putchar(ramp[level]);
        }
        putchar('\n');
    }
}

/* Read up to 784 numbers (any whitespace/comma/bracket separated) and apply
 * the requested normalisation into img[1][28][28]. */
static bool load_image(const std::string &path, const std::string &fmt,
                       float img[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH]) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open image file: " << path << "\n"; return false; }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    for (char &c : s) if (c == ',' || c == '[' || c == ']' || c == '{' || c == '}' || c == ';') c = ' ';
    std::stringstream ns(s);
    std::vector<float> vals; float x;
    while (ns >> x) vals.push_back(x);
    const int N = IMAGE_HEIGHT * IMAGE_WIDTH;
    if ((int)vals.size() < N) {
        std::cerr << "Expected " << N << " values, got " << vals.size() << "\n";
        return false;
    }
    for (int h = 0; h < IMAGE_HEIGHT; h++)
        for (int w = 0; w < IMAGE_WIDTH; w++) {
            float p = vals[h * IMAGE_WIDTH + w];
            float n;
            if      (fmt == "std")  n = p;                                  // already normalised
            else if (fmt == "unit") n = (p - MNIST_MEAN) / MNIST_STD;       // 0..1 grayscale
            else                    n = (p / 255.0f - MNIST_MEAN) / MNIST_STD; // raw 0..255 (default)
            img[0][h][w] = n;
        }
    return true;
}

int main(int argc, char **argv) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id",   "-d", "device index", "0");
    parser.addSwitch("--image",       "-i", "custom 28x28 image file (784 values)", "");
    parser.addSwitch("--format",      "-f", "custom image format: raw|unit|std", "raw");
    parser.addSwitch("--num",         "-n", "number of MNIST images to test", "100");
    parser.addSwitch("--label",       "-l", "true label for the custom image (optional)", "-1");
    parser.parse(argc, argv);

    std::string binaryFile = parser.value("xclbin_file");
    int device_index       = stoi(parser.value("device_id"));
    std::string imgFile    = parser.value("image");
    std::string fmt        = parser.value("format");
    int num                = stoi(parser.value("num"));
    int userLabel          = stoi(parser.value("label"));
    if (binaryFile.empty()) { parser.printHelp(); return EXIT_FAILURE; }
    if (num > NUM_IMAGES) num = NUM_IMAGES;

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    auto krnl = xrt::kernel(device, uuid, "cnn");

    // Kernel args: (0) input_image [1][28][28], (1) classification [10].
    auto bo_in  = xrt::bo(device, sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH, krnl.group_id(0));
    auto bo_out = xrt::bo(device, sizeof(float) * FC2_FEATURES,               krnl.group_id(1));
    auto p_in   = bo_in.map<float *>();
    auto p_out  = bo_out.map<float *>();

    auto run_fpga = [&](const float img[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH], float logits[FC2_FEATURES]) {
        for (int h = 0; h < IMAGE_HEIGHT; h++)
            for (int w = 0; w < IMAGE_WIDTH; w++)
                p_in[h * IMAGE_WIDTH + w] = img[0][h][w];
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = krnl(bo_in, bo_out);
        r.wait();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        for (int i = 0; i < FC2_FEATURES; i++) logits[i] = p_out[i];
    };

    /* ---------------- Mode 2: custom image ---------------- */
    if (!imgFile.empty()) {
        static float img[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH];
        if (!load_image(imgFile, fmt, img)) return EXIT_FAILURE;

        float logits[FC2_FEATURES], probs[FC2_FEATURES];
        run_fpga(img, logits);
        softmax10(logits, probs);
        int pred = argmax10(logits);

        std::cout << "\nInput image (" << fmt << " format):\n";
        print_ascii(img);
        std::cout << "\n digit :   logit    probability\n";
        std::cout << "-------------------------------\n";
        for (int i = 0; i < FC2_FEATURES; i++)
            printf("   %d   : %8.3f   %6.2f%%%s\n", i, logits[i], probs[i] * 100.0f,
                   (i == pred) ? "   <== predicted" : "");
        printf("\nPredicted digit: %d\n", pred);
        if (userLabel >= 0)
            std::cout << (pred == userLabel ? "CORRECT" : "WRONG")
                      << " (expected " << userLabel << ")\n";
        return 0;
    }

    /* ---------------- Mode 1: MNIST accuracy + benchmark ---------------- */
    std::cout << "\nClassifying " << num << " MNIST test images on the FPGA...\n";
    int correct = 0;
    float logits[FC2_FEATURES];
    for (int i = 0; i < num; i++) {
        run_fpga(INPUT_IMAGES[i], logits);
        int pred = argmax10(logits);
        if (pred == res_cnn_real[i]) correct++;
        printf("  image %3d: predicted %d, actual %d  %s\n",
               i, pred, res_cnn_real[i], (pred == res_cnn_real[i]) ? "" : "  <-- miss");
    }
    printf("\nFPGA accuracy: %d/%d = %.2f%%\n", correct, num, 100.0f * correct / num);

    // Timing: FPGA vs ARM software (same cnn() compiled for the A53).
    static float sw_logits[FC2_FEATURES];
    double fpga_total = 0.0, arm_total = 0.0;
    for (int i = 0; i < num; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_fpga(INPUT_IMAGES[i], logits);
        auto t1 = std::chrono::high_resolution_clock::now();
        fpga_total += std::chrono::duration<double>(t1 - t0).count();

        auto s0 = std::chrono::high_resolution_clock::now();
        cnn(INPUT_IMAGES[i], sw_logits);           // ARM software reference
        auto s1 = std::chrono::high_resolution_clock::now();
        arm_total += std::chrono::duration<double>(s1 - s0).count();
    }
    std::cout << "\n--- Performance (avg over " << num << " images) ---\n";
    printf("FPGA time : %0.3f us/image\n", fpga_total / num * 1e6);
    printf("ARM  time : %0.3f us/image\n", arm_total  / num * 1e6);
    printf("Speedup   : %0.2fx\n",         arm_total / fpga_total);
    return 0;
}
