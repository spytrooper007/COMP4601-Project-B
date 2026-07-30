# LeNet-5 — full CNN for MNIST (UNOPTIMISED baseline)

Full LeNet-5-style CNN (2 conv + 2 fully-connected layers) classifying MNIST
digits. **No performance pragmas** — sequential baseline. Verified: **100 % on
the bundled 100-image MNIST test set** (C-sim and on the KV260).

## Source / attribution
Ported from **Lenet5FloatHLS** — David de Andrés & Juan Carlos Ruiz, Universitat
Politècnica de València, <https://git.upv.es/defadas/Lenet5FloatHLS>
(commit `6277ee1`, MIT). Copyright headers and `../LICENSE.txt` preserved.
One local fix: an include casing (`cnn_pad1_res.h` → `cnn_Pad1_res.h`) so it
builds on case-sensitive Linux/Vitis.

## Folder layout — what each part is for

| Path | Component | Role |
|------|-----------|------|
| `hls/src/core/`   | kernel | the CNN layers + `cnn.cpp` (HLS top). Only `cnn.cpp` keeps AXI `INTERFACE` pragmas |
| `hls/include/core/` | kernel | layer headers, `cnn_defines.h`, and the **trained weights** (`cnn_weights_and_bias_all_layers.h`) |
| `hls/src/test/` + `hls/include/test/` | testbench | `cnntest.cpp` (main) + per-layer/whole-net tests + `oracle/` golden vectors + MNIST images/labels |
| `lenet5_hls/`     | HLS component | `vitis-comp.json` + `hls_config.cfg` (top = `cnn`, part, 200 MHz, source list) |
| `lenet5_system/`  | system component | `vitis-sys.json` (links the kernel → bitstream) |
| `lenet5_host/`    | host | `lenet5_host.cpp` (XRT, MNIST accuracy + custom-image inference) + `cmdlineparser`/`logger` + `CMakeLists.txt`/`UserConfig.cmake`/`vitis-comp.json` |
| `sample_images/`  | data | example 28×28 inputs (a "7") for the host's `-i` mode |

## The weights
Already trained and **compiled into the kernel** (`static const` arrays in
`hls/include/core/cnn_weights_and_bias_all_layers.h`) — nothing is passed at
runtime; they end up in the bitstream. Eight arrays: `KERNEL_CONV_1[3][1][5][5]`,
`BIAS_CONV_1[3]`, `KERNEL_CONV_2[6][3][5][5]`, `BIAS_CONV_2[6]`,
`WEIGHTS_FC1[147][294]`, `BIAS_FC1[147]`, `WEIGHTS_FC2[10][147]`, `BIAS_FC2[10]`.

## Build & run C-simulation (host PC, no board)
```bash
cd hls
g++ -O2 -w -std=c++14 -Iinclude/core -Iinclude/test -Iinclude/test/oracle \
    src/core/*.cpp src/test/*.cpp -o csim
./csim 0 0 100      # classify images 0..99, prints accuracy
```

## On the KV260 (host)
After building/packaging to `lenet5.bin` and building `lenet5_host`:
```bash
./lenet5_host -x lenet5.bin                         # 100-image MNIST accuracy + FPGA-vs-ARM timing
./lenet5_host -x lenet5.bin -i sample_images/digit7_raw.txt   # classify your own 28x28 image
```
Custom image = 784 numbers (row-major). `--format raw` (0..255, default) applies
MNIST normalization `(p/255 − 0.1307)/0.3081`; `unit` (0..1); `std` (already
normalized). Feeding raw pixels without `raw` gives garbage — the model expects
standardized input (background ≈ −0.4242).
