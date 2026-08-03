# How to accelerate this workspace further (maximise LUT/DSP usage)

The conv layers are already fast (pipelined, unrolled window, fixed-point). But
the network still runs **layer-by-layer, sequentially**, and the **fully-connected
layers are still the sequential baseline** — so FC1 is now the bottleneck, not the
convolutions. Work the list below top-to-bottom; after each change run
C-Synthesis and read the **utilisation** (LUT/DSP/BRAM %) and **timing** (II,
latency, slack at 5 ns) reports.

## Where the time actually goes now (measure first)
Rough cycle counts @200 MHz, current design:

| Layer | ~iterations | note |
|-------|-------------|------|
| conv1 | 3×28×28 = 2352 | pipelined, ~fast |
| conv2 | 6×14×14 = 1176 | pipelined, ~fast |
| **FC1** | **147×294 = 43 218 MACs, sequential** | **dominant — fix this first** |
| FC2 | 147×10 = 1470 | sequential |
| pool/pad/flatten | small | sequential |

**So the #1 win is the FC layers, not more conv tuning.**

---

## Priority 1 — Accelerate FC1/FC2 (biggest latency win)
Convert them to fixed-point and parallelise the inner MAC loop. In
`fullyConnected_1.cpp` / `fullyConnected_2.cpp`:

```c
// pipeline the output-neuron loop, unroll the reduction by a factor
fc1_F: for (f = 0; f < FC1_FEATURES; f++) {
#pragma HLS PIPELINE II=1
    acc_t acc = 0;
    fc1_NIF: for (nif = 0; nif < FC1_INPUT_FEATURES; nif++) {
#pragma HLS UNROLL factor=16          // 16 parallel MACs; raise until you run out of LUT/DSP
        acc += (acc_t)in[nif] * (acc_t)w[f][nif];
    }
    out[f] = acc + bias[f];
}
```
and partition the operands so 16 lanes can read in one cycle:
```c
#pragma HLS ARRAY_PARTITION variable=input_weights  cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=input_features cyclic factor=16 dim=1
```
- Start `factor=8`, synthesize, check utilisation, then push to 16/32.
- Fixed-point (`ap_fixed<20,8>` like the conv) makes each MAC cheap, so you can
  unroll far more than the float version could (that's what overflowed before).

## Priority 2 — Unroll the conv output-channel loop
Right now each conv does one output map at a time. Unroll `conv_K` to do all maps
in parallel (conv1 ×3, conv2 ×6):
```c
conv1_K: for (int k = 0; k < CONV1_KERNELS; k++) {
#pragma HLS UNROLL                     // 3x the conv1 hardware
```
and add `#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1` so every
kernel's weights are available at once. This multiplies conv throughput and LUT/DSP
use by the channel count.

## Priority 3 — Remove the input-buffer port bottleneck (real II=1)
`in_buf` is only partitioned on the channel dim, so the 25/75 window reads fight
over the same BRAM ports and the pipeline can't truly hit II=1. Partition the row
dimension so the window reads in parallel:
```c
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=5 dim=2   // = kernel height
```
(or `complete dim=2 dim=3` for small feature maps). Check the synthesis "II" — it
should drop to 1.

## Priority 4 — Pipeline the cheap layers
Add `#pragma HLS PIPELINE II=1` to the innermost loops of `padding_1/2`,
`maxPool_1/2`, and `flattenLayer`. Small latency, near-free — removes the
sequential tails.

## Priority 5 — Overlap layers with DATAFLOW (throughput)
At the top of `cnn.cpp`:
```c
#pragma HLS DATAFLOW
```
so layer N+1 starts consuming while layer N is still producing. Biggest win when
classifying a *stream* of images. For best results convert the inter-layer arrays
to `hls::stream<>` (line-by-line producer/consumer). This is the largest
architectural change — do it last.

---

## To literally "max out the LUTs" (and stop DSPs being the limiter)
The KV260 has **1248 DSPs but 117 120 LUTs** — DSPs run out first. So push the
multipliers into **LUT fabric** and keep the DSPs for where they help:
```c
#pragma HLS BIND_OP variable=acc op=mul impl=fabric   // multiplier in LUTs, not DSP
```
Combined with **smaller fixed-point** (try `ap_fixed<16,6>`, or `<12,4>` if
accuracy holds — verify with co-sim!), each LUT multiplier shrinks, so you can
unroll far more before running out. This is the knob that turns "spare LUTs" into
more parallelism.

## The method (don't guess — iterate)
1. Change one thing (raise an unroll factor, add a partition).
2. Run **C-Synthesis**, open the report.
3. Look at: **Total LUT/DSP/BRAM %**, per-loop **II** and **latency**, and
   **timing slack at 5 ns**.
4. Keep increasing parallelism until you hit **~75–85 % of LUT or DSP**, or until
   **timing stops closing at 200 MHz** — whichever comes first. Leave ~15–20 %
   headroom or place-and-route will fail (congestion), like the earlier overflow.
5. Only then run the full system build (30-min P&R) and deploy.

## Accuracy guardrail
Every precision change (`ap_fixed` widths, `impl=fabric`) can shift results.
Run **C/RTL co-simulation** (or an on-board accuracy pass) after shrinking
precision — if the 100 % MNIST accuracy drops, add fractional bits back.
