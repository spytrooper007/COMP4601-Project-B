# LeNet acceleration — measured results

C-synthesis experiments on the full network, Vitis HLS 2025.2, target
**KV260 `xck26-sfvc784-2LV-c` @ 200 MHz (5 ns)**. Each variant is cumulative.
Latency = C-synthesis estimate for one image; accuracy = MNIST 100-image test
set. "Fixed" accuracy = `ap_fixed<20,8>` datapath (emulated in software against
the same test set).

## Summary table

| # | Variant | Latency (cyc) | µs @200 MHz | Speedup vs V0 | LUT | DSP | FF | BRAM | Timing | Accuracy (float / fixed) | Fits KV260? |
|---|---------|--------------:|------------:|--------------:|----:|----:|---:|-----:|:------:|:--------------:|:-----------:|
| V0 | Baseline (no explicit pragmas) | 1,202,163 | 6010.8 | 1.0× | 25,611 (21%) | 136 (10%) | 26,409 (11%) | 136 (47%) | met | 100% / — | ✓ |
| V1 | + accelerated **conv** (fixed-point, adder tree) | 411,347 | 2056.7 | **2.9×** | 43,236 (36%) | 219 (17%) | 20,968 (8%) | 112 (38%) | met | 100% / 100% | ✓ |
| V2 | + accelerated **FC** (8 MACs/cyc) | 16,629 | 83.1 | **72.3×** | 68,005 (58%) | 245 (19%) | 26,606 (11%) | 152 (52%) | met | 100% / 100% | ✓ |
| V3 | + **conv channel-parallel** (max unroll) | 14,080 | 70.4 | **85.4×** | 127,998 (**109%**) | 539 (43%) | 36,308 (15%) | 176 (61%) | met* | 100% / 100% | **✗ LUT over 100%** |

\* C-synthesis reports timing met, but at 109 % LUT the design **cannot be
placed-and-routed** — implementation would fail (same failure mode as the early
float-unrolled attempt).

## Feeding the real bottleneck: FC unroll factor sweep (on top of V2)
Instead of V3's wasted conv parallelism, raise the FC unroll factor (FC is 44 %
of V2's time):

| FC unroll | Latency (cyc) | µs | Speedup vs V0 | LUT | DSP | Fits? |
|-----------|--------------:|---:|--------------:|----:|----:|:-----:|
| ×8  (= V2) | 16,629 | 83.1 | 72× | 58% | 19% | ✓ |
| **×16 ⭐** | **13,893** | **69.5** | **87×** | **77%** | 22% | **✓ (best deployable)** |
| ×32 | 12,520 | 62.6 | 96× | **114%** | 27% | ✗ over LUT |

**FC ×16 is the winner: 69.5 µs, fits at 77 % LUT.** It is *faster than the
channel-parallel V3 (70.4 µs) and actually fits* (V3 was 109 %) — proving the
point: spend the LUTs on the bottleneck (FC), not the conv. ×32 is faster still
but overflows. Accuracy stays 100 % (float and `ap_fixed<20,8>`) at every factor.

**→ The `lenet5_full_accel` workspace now ships this FC ×16 design.**

## What each step did (and why)

**V0 → V1 (accelerate conv): only 2.9×.**
The baseline's two bottlenecks were `conv2` and `fc1`, both stalled at **II = 9**
(memory-dependency). The fixed-point + adder-tree conv drops **conv2 II 9 → 1**
(794k → a few k cycles). But `fc1` was still the sequential baseline at
**388,979 cycles**, so it now dominated — capping the overall gain at 2.9×.
*Lesson: accelerating the conv alone barely moves the needle, because the conv
was never the bottleneck.*

**V1 → V2 (accelerate FC): jumps to 72×.**
Making FC1/FC2 fixed-point with 8 parallel MACs/cycle drops **fc1 388,979 →
7,351 cycles (53×)**. This is the single biggest win in the whole study.
LUT rises to 58 % — comfortably within budget. **V2 is the sweet spot.**

**V2 → V3 (max-out conv parallelism): +1.18× for +51 % LUT — a bad trade.**
Unrolling conv fully across output channels (450 fixed-point MACs/cycle in conv2)
pushes **LUT to 109 % — over the device** — while only improving latency from
83 → 70 µs. Because conv is already tiny after V2, the extra hardware buys almost
nothing. *Lesson: "maxing out the LUTs" is only worth it on the actual
bottleneck.*

## Where the time goes in V2 (the new profile)

| Block | cycles | % |
|-------|-------:|--:|
| **fc1** | 7,351 | 44 % |
| conv1 (incl. buffer load) | 3,407 | 20 % |
| conv2 (incl. buffer load) | 2,177 | 13 % |
| pad1 / pad2 | 2,074 | 12 % |
| pool1 / pool2 / flatten | 1,204 | 7 % |
| fc2 | 399 | 2 % |

## To go faster than V2 *within* the LUT budget
Aim at the bottleneck, not the conv:
1. **Raise the FC1 unroll factor** (8 → 16 → 32). FC1 is 44 % of the remaining
   time; this is the highest-value knob and FC is cheap in fixed-point.
2. **`#pragma HLS DATAFLOW`** at the top of `cnn.cpp` to overlap layers and hide
   the ~4,000 cycles of buffer-load + padding overhead now visible.
3. **Leave the conv alone** — V3 proved extra conv parallelism is wasted LUT.

## Accuracy
- Every variant classifies the 100-image MNIST set at **100 %** in float C-sim.
- **Fixed-point `ap_fixed<20,8>` also gives 100 %** (verified for V2; V1/V3 use
  identical datatypes and math), so the speedups cost no accuracy. If you shrink
  precision further to save LUTs, re-check this.

## How these were produced
Full-network C-synthesis via `v++ -c --mode hls` (top = `cnn`), one config per
variant; utilisation/latency read from `hls/syn/report/csynth.rpt`; accuracy via
the `cnntest` testbench compiled with g++ (float) and with the `ap_fixed` branch
forced against Vitis headers (fixed-point).
