# COMP4601 Project B — Unoptimised CNN Accelerators (KV260)

Baseline (un-accelerated) HLS implementations of the two kernels for the
*Accelerating a CNN on the Kria KV260* project. These are the **reference points**
the optimised versions are measured against — deliberately sequential, with **no
performance pragmas** (no `PIPELINE`, `UNROLL`, or `ARRAY_PARTITION`; only the
required AXI `INTERFACE` pragmas remain).

## Two components

| Folder | What | Status |
|--------|------|--------|
| [`conv5x5_unoptimised/`](conv5x5_unoptimised) | Single 5×5 convolution + ReLU accelerator (LeNet Conv1 dimensions: 1×32×32 → 6×28×28) | baseline, verified |
| [`lenet5_unoptimised/`](lenet5_unoptimised)   | Full LeNet-5 CNN for MNIST (2 conv + 2 FC) | baseline, verified 100% on 100 MNIST images |

Each folder has its own `README.md` listing exactly which file does what
(kernel, testbench, host, config).

## Other files
- `docs/COMP4601 Project Slides.pdf` — the project brief.
- `LICENSE.txt` — MIT (the LeNet-5 core is ported from Universitat Politècnica de
  València; attribution in `lenet5_unoptimised/README.md`).
- `UPDATES.md` — running log of scope/direction changes.

## Quick verify on a host PC (no board, no Vitis)
Both kernels are plain C++ (HLS pragmas are ignored by g++), so C-simulation
runs anywhere:
```bash
# conv5x5 baseline
cd conv5x5_unoptimised && g++ -O2 -std=c++14 conv5x5.cpp conv5x5_test.cpp -o csim && ./csim

# lenet5 baseline (classify 100 MNIST images)
cd ../lenet5_unoptimised/hls && g++ -O2 -w -std=c++14 \
    -Iinclude/core -Iinclude/test -Iinclude/test/oracle \
    src/core/*.cpp src/test/*.cpp -o csim && ./csim 0 0 100
```

## Relationship to the optimised version
The optimised build re-adds `PIPELINE`/`UNROLL`/`ARRAY_PARTITION` (tuned to fit
the KV260's DSP/LUT budget). Comparing the two — plus the ARM software time the
host reports — is the point of the project.
