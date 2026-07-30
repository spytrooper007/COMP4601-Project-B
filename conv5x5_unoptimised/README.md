# conv5x5 — single 5×5 convolution accelerator (UNOPTIMISED baseline)

One 5×5 convolution + ReLU layer, sized to LeNet-5's Conv1: input `1×32×32`
(a 28×28 image zero-padded by 2), `OUT_CH=6` kernels of `5×5`, output `6×28×28`.
`float` datatype. **No performance pragmas** — sequential baseline.

## Files — what each is for

| File | Role |
|------|------|
| `conv5x5.h`        | dimensions + kernel prototype |
| `conv5x5.cpp`      | the HLS **kernel** (hardware) — has only the AXI `INTERFACE` pragmas |
| `conv5x5_test.cpp` | C-simulation **testbench** — builds a golden reference in plain C and checks RMSE (self-contained, no data files) |
| `conv5x5_host.cpp` | **XRT host** for the KV260 PS — feeds the kernel and times FPGA vs ARM |

## Build & run C-simulation (host PC, no board)
```bash
g++ -O2 -std=c++14 conv5x5.cpp conv5x5_test.cpp -o csim
./csim        # prints RMSE + PASS
```
Verified PASS (RMSE ≈ 2e-8, float kernel vs double golden).

## In Vitis HLS
Create an HLS component with top function `conv5x5`, source `conv5x5.cpp`,
testbench `conv5x5_test.cpp`, part `xck26-sfvc784-2LV-c`, 200 MHz. Because there
are no acceleration pragmas, it uses minimal DSP/LUT and closes timing easily —
that's the baseline latency you compare the optimised kernel against.

## On the KV260
Build `conv5x5_host.cpp` against XRT, package the kernel to `conv5x5.bin`, then:
```bash
./conv5x5_host -x conv5x5.bin
```
Prints the correctness check and the average FPGA vs ARM latency.
