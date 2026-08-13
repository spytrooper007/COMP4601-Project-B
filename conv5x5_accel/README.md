# conv5x5 Accelerator — Vitis Host Flow

Standalone 5×5 convolution + ReLU kernel accelerated on the Kria KV260 FPGA
using Vitis HLS and XRT. The host program (`conv5x5_host.cpp`) loads the
compiled bitstream, transfers data to the device, runs the kernel, and checks
the FPGA output against a plain-C golden reference before reporting a speedup
over an ARM software baseline.

---

## What this accelerator does

A single-layer, single-channel 5×5 convolution with ReLU activation:

| Parameter | Value |
|-----------|-------|
| Input     | 1 channel × 32 × 32 (1024 floats) |
| Kernel    | 5×5, 6 output channels (150 weights + 6 biases) |
| Output    | 6 channels × 28 × 28 (4704 floats) |
| Activation | ReLU (in-kernel) |

The kernel (`conv5x5.cpp`) uses fixed-point types (`ap_fixed<20,8>` for data,
`ap_fixed<40,16>` for accumulators) during synthesis, with `m_axi` interfaces
for the four float buffers and an `s_axilite` control interface. On the CPU
side (testbench and host), all types fall back to `float`.

---

## File layout

```
conv5x5_accel/
├── conv5x5.h              # Dimension macros + type aliases (float / ap_fixed)
├── conv5x5.cpp            # HLS kernel — top function conv5x5()
├── conv5x5_test.cpp       # C-simulation testbench (golden comparison, RMSE)
├── conv5x5_host.cpp       # XRT host program (correctness + timing vs ARM)
├── hls_config.cfg         # Vitis HLS component config (top, sources, -I paths)
├── vitis-comp.json        # HLS component descriptor
├── vitis-sys.json         # System (hw_link + package) descriptor
└── lenet5_host/           # Vitis host component (CMakeLists, cmdlineparser, …)
```

---

## Step 1 — C-simulation (no board needed)

Compile the testbench directly with g++ to verify correctness before touching
Vitis:

```bash
cd conv5x5_accel
g++ -O2 -std=c++14 -o conv5x5_csim conv5x5.cpp conv5x5_test.cpp
./conv5x5_csim
```

Expected output:

```
----------------------------------------------
   Conv5x5  RMSE
0.000000000000000
----------------------------------------------
*******************************************
PASS: The output matches the golden output!
*******************************************
```

The testbench generates a fixed, deterministic input (`(i%17 − 8) × 0.1`),
weight set (`(i%7 − 3) × 0.05`), and bias (`(i − 3) × 0.02`), computes a
double-precision golden reference, runs the HLS function, and checks that the
RMSE is below `1e-3`. The same data is used by the host program so the two
comparisons are consistent.

---

## Step 2 — Open in Vitis and run C-simulation

This folder is a Vitis workspace. Open it directly:

```
File ▸ Open Workspace… ▸ conv5x5_accel/
```

or from the terminal:

```bash
vitis -w conv5x5_accel/
```

Three components are pre-configured:

| Component | Type | Notes |
|-----------|------|-------|
| `conv5x5_hls` | HLS | top = `conv5x5`; `conv5x5.cpp` as synth source, `conv5x5_test.cpp` as testbench |
| `conv5x5_system` | System (hw_link + package) | `nk=conv5x5:1:conv5x5_1` |
| `lenet5_host` | Host (ARM / XRT) | Correctness check + timing benchmark |

To run C-simulation in Vitis: select **`conv5x5_hls`** → *Run ▸ C Simulation*.
This calls `conv5x5_test.cpp::main()` and prints the same RMSE / PASS block as
above.

---

## Step 3 — Synthesise and build the bitstream

In the Vitis GUI:

1. Select **`conv5x5_hls`** → *Run ▸ C Synthesis* (targets the KV260 / XCK26 at 300 MHz by default in `hls_config.cfg`).
2. Select **`conv5x5_system`** → *Build* to run `v++` link and package. This produces `conv5x5.bin` and the associated `pl.dtbo` / `shell.json`.

Or from the command line (after sourcing the Vitis environment):

```bash
vitis -s conv5x5_accel/hls_config.cfg   # synthesis only
```

---

## Step 4 — Deploy to the KV260

Copy the packaged output to the board and load the overlay:

```bash
# On the host PC — copy files to the board
scp conv5x5.bin pl.dtbo shell.json petalinux@<board-ip>:~/

# On the board
sudo mkdir -p /lib/firmware/xilinx/conv5x5
sudo cp conv5x5.bin pl.dtbo shell.json /lib/firmware/xilinx/conv5x5/
sudo xmutil unloadapp
sudo xmutil loadapp conv5x5
```

---

## Step 5 — Build and run the host program

The host is built by Vitis as part of the `lenet5_host` component. To build
manually on the board (or cross-compile with the sysroot):

```bash
# On the KV260 directly (XRT already installed)
g++ -std=c++14 -O2 \
    -I/usr/include/xrt \
    -o conv5x5_host \
    conv5x5_host.cpp lenet5_host/cmdlineparser.cpp \
    -lxrt_coreutil -lpthread -lrt -lstdc++
```

Run:

```bash
./conv5x5_host -x conv5x5.bin
```

---

## What the host program checks

### Correctness vs golden output

The host generates the same deterministic test vectors as `conv5x5_test.cpp`
and computes the golden output on the ARM CPU using `conv5x5_sw()` (a plain-C
double-loop convolution). After the FPGA run it compares element-by-element and
reports the RMSE:

```
----------------------------------------------
Conv5x5 RMSE : 0.000000000000000
PASS: matches golden output
----------------------------------------------
```

A result is considered correct if RMSE < `1e-3`. The fixed-point types used in
synthesis (`ap_fixed<20,8>` / `ap_fixed<40,16>`) introduce small quantisation
errors but stay well within this threshold.

### Performance vs ARM software baseline

The host then runs both the FPGA kernel and the ARM software convolution 1000
times each and reports average latency and speedup:

```
--- Performance (avg over 1000 runs) ---
FPGA time :  XX.XXX us
ARM  time : XXX.XXX us
Speedup   : X.XXx
```

The FPGA timing covers the full XRT round-trip: `sync to device → kernel
execution → sync from device`. The ARM timing runs the same plain-C reference
loop. The speedup figure is `ARM avg / FPGA avg`.

---

## Key HLS optimisations in conv5x5.cpp

| Pragma | Effect |
|--------|--------|
| `PIPELINE II=1` on `conv_OW` | One output pixel initiated per clock cycle |
| `ARRAY_PARTITION complete` on `w_buf` (dim 3, 4) | All 25 kernel weights available in parallel |
| `ARRAY_PARTITION complete` on `in_buf` (dim 1) | All input channels read simultaneously |
| Tree-reduction of 25 products into a 32-element balanced adder tree | Reduces accumulation depth, improves timing |
| Separate `gmem0 / gmem1 / gmem2` AXI bundles | Input, weights+bias, and output use independent memory ports — no bank conflicts |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FAIL: does not match golden output` | Bitstream is stale or wrong `.bin` loaded | Rebuild `conv5x5_system`; confirm `xmutil loadapp conv5x5` succeeded |
| `xclbin` open fails | Wrong device index or overlay not loaded | Run `sudo xmutil loadapp conv5x5` first; use `-d 0` |
| RMSE > 0 but < 1e-3 | Expected — fixed-point quantisation | Normal; the threshold is intentionally generous |
| Very low speedup | XRT transfer overhead dominates for this small kernel size | Expected for a single-layer benchmark; full-network speedup (see `lenet5_unoptimised`) is more representative |
