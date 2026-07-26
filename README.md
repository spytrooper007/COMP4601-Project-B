# LeNet-5 (float) — Vitis HLS

Full LeNet-5-style CNN for MNIST in C++ for High-Level Synthesis. This is the
"extend to the full network" target of the project; the standalone single-layer
accelerator lives in [`../conv5x5`](../conv5x5).

## Source / attribution

Ported from the reference implementation:

- **Lenet5FloatHLS** — David de Andrés & Juan Carlos Ruiz, Fault-Tolerant
  Systems, Instituto ITACA, Universitat Politècnica de València.
  <https://git.upv.es/defadas/Lenet5FloatHLS> (commit `6277ee1`), MIT License.

The original targets the Zynq UltraScale+ ZCU104 (XCZU7EV) with Vitis HLS 2023.1
— the same UltraScale+ family as the Kria KV260 (XCK26), so the HLS flow carries
over. All original copyright headers and `LICENSE.txt` are preserved.

### Local changes from upstream
- `hls/include/test/test_padding_1.h`: fixed the include `oracle/cnn_pad1_res.h`
  → `oracle/cnn_Pad1_res.h`. Upstream was developed on a case-insensitive
  filesystem; Linux (and Vitis on the KV260) is case-sensitive and would not
  compile without this.

## Architecture

Input `1×28×28`, zero-padded to `32×32`. A reduced LeNet-5 (fewer feature maps
than the classic 6/16/120 to keep it small, as the slides plan):

| Layer      | Op                        | Output    |
|------------|---------------------------|-----------|
| padding_1  | zero-pad 2                | 1×32×32   |
| conv1+ReLU | 3 kernels, 5×5            | 3×28×28   |
| maxPool_1  | 2×2                       | 3×14×14   |
| padding_2  | zero-pad 2                | 3×18×18   |
| conv2+ReLU | 6 kernels, 5×5            | 6×14×14   |
| maxPool_2  | 2×2                       | 6×7×7     |
| flatten    | —                         | 294       |
| fc1        | 294 → 147                 | 147       |
| fc2        | 147 → 10                  | 10 logits |

`hls/src/core/cnn.cpp` is the HLS top with `s_axilite`/`m_axi` interfaces —
this is the **hardware** kernel. The same C++ compiled for the CPU is the
**software** baseline.

## Layout

```
hls/include/core/   layer headers, cnn_defines.h, trained weights & biases, input images
hls/include/test/   testbench headers + oracle/ per-layer golden vectors
hls/src/core/       layer implementations (padding, conv, maxpool, flatten, FC) + cnn.cpp top
hls/src/test/       cnntest.cpp (main) + per-layer + whole-network tests
```

## Build & run C-simulation (host CPU, no board needed)

```bash
cd hls
g++ -O2 -w -std=c++14 \
    -Iinclude/core -Iinclude/test -Iinclude/test/oracle \
    src/core/*.cpp src/test/*.cpp -o lenet5_csim

./lenet5_csim <op> <from_image> <to_image>
#   op 0 = whole CNN classification, 1..9 = individual layers vs oracle
./lenet5_csim 0 0 100      # classify images 0..99, print accuracy
```

Verified: `./lenet5_csim 0 0 20` classifies the first 20 MNIST images correctly
(100%).

## Open in Vitis & run C-sim

This folder is a Vitis workspace. Open it directly:
`File ▸ Open Workspace… ▸ Project/lenet5`  (or `vitis -w Project/lenet5`).

Three components are pre-configured (mirrors dft256 / conv5x5):

| Component        | Type       | Notes |
|------------------|------------|-------|
| `lenet5_hls`     | HLS        | top = `cnn`; all `hls/src/core/*.cpp` as synth sources, `hls/src/test/*.cpp` as testbench, `-I` paths set in `hls_config.cfg` |
| `lenet5_system`  | System (hw_link + package) | `nk=cnn:1:cnn_1` |
| `lenet5_host`    | Host (ARM/XRT) | MNIST accuracy + custom-image inference |

**To run C-simulation:** select **`lenet5_hls`** → *Run ▸ C Simulation*. With no
testbench args it classifies MNIST image 0 (`test_cnn`, op 0). To sweep a range,
set testbench arguments `0 0 100` in the component's run settings (op, from, to).

> Open `Project/lenet5` itself as the workspace (not the parent `Project/`).

## The weights — what to use

**You don't pass weights at runtime.** The trained parameters are compiled
*into* the kernel as `static const` arrays in
[`hls/include/core/cnn_weights_and_bias_all_layers.h`](hls/include/core/cnn_weights_and_bias_all_layers.h),
so they end up baked into the bitstream. The host only sends the 28×28 image and
reads back 10 logits. The eight arrays (from the UPV reference, PyTorch-trained
on MNIST) are:

| Array            | Shape            | Layer |
|------------------|------------------|-------|
| `KERNEL_CONV_1`  | `[3][1][5][5]`   | conv1 weights |
| `BIAS_CONV_1`    | `[3]`            | conv1 bias |
| `KERNEL_CONV_2`  | `[6][3][5][5]`   | conv2 weights |
| `BIAS_CONV_2`    | `[6]`            | conv2 bias |
| `WEIGHTS_FC1`    | `[147][294]`     | fc1 weights |
| `BIAS_FC1`       | `[147]`          | fc1 bias |
| `WEIGHTS_FC2`    | `[10][147]`      | fc2 weights |
| `BIAS_FC2`       | `[10]`           | fc2 bias |

To use **your own** weights, retrain the same architecture (see the layer shapes
in the table above / `cnn_defines.h`) and regenerate that one header with the
identical array names and shapes — nothing else changes. The current weights
already give 100 % on the bundled 100-image MNIST test set.

## Run on the KV260 (host)

Build the `lenet5_system` (→ `lenet5.bin`) and `lenet5_host`, deploy like dft256
(copy `lenet5.bin` / `pl.dtbo` / `shell.json` to `/lib/firmware/xilinx/lenet5/`,
`sudo xmutil loadapp lenet5`), then:

### MNIST accuracy test
```bash
./lenet5_host -x lenet5.bin            # classify 100 MNIST images, print accuracy
./lenet5_host -x lenet5.bin -n 50      # first 50 images
```
Prints per-image `predicted vs actual`, an overall accuracy %, and an
FPGA-vs-ARM timing/speedup block (labels come from `res_cnn_real` in
`mnist_labels_and_predictions.h`).

### Classify your own image
```bash
./lenet5_host -x lenet5.bin -i sample_images/digit7_raw.txt              # raw 0..255 (default)
./lenet5_host -x lenet5.bin -i sample_images/digit7_std.txt -f std       # already normalized
./lenet5_host -x lenet5.bin -i mydigit.txt -f raw -l 7                   # check against label 7
```
The image file is **784 numbers** (28×28, row-major, any whitespace/comma
separated). `--format`:
- `raw`  (default) — pixels `0..255`, host applies MNIST normalization `(p/255 − 0.1307) / 0.3081`
- `unit` — pixels `0..1`
- `std`  — values already standardized (pass-through)

Output: an ASCII preview of your image, the 10 logits + soft-max probabilities,
and the predicted digit. Two ready examples are in
[`sample_images/`](sample_images) (image 0, a "7"), which you can also use as a
template for your own.

> **Normalization matters:** the model was trained on standardized MNIST
> (background = −0.4242). If you feed raw pixels without `--format raw`, the
> prediction will be garbage. `sample_images/digit7_raw.txt` +
> `-f raw` and `sample_images/digit7_std.txt` + `-f std` were both verified to
> predict 7.

Both sample files, the normalization, and the accuracy path are verified through
the software network on the host CPU; the XRT plumbing mirrors the working
dft256 / conv5x5 hosts.
