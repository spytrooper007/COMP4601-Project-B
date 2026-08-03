# COMP4601 Project B — CNN Acceleration on the Kria KV260

HLS implementations of a **LeNet-inspired CNN** (2 conv + 2 FC) for MNIST, taken
from an unoptimised baseline through progressive hardware acceleration on the
KV260 (`xck26`, 200 MHz). The point of the project is the **acceleration
methodology** — the CNN is the representative workload.

## Workspaces (the acceleration progression)

| Folder | What | Latency¹ | Speedup | LUT | Accuracy |
|--------|------|---------:|--------:|----:|:--------:|
| [`lenet5_unoptimised/`](lenet5_unoptimised) | Baseline — sequential, no perf pragmas | 6011 µs | 1.0× | 21% | 100% |
| [`lenet5_conv_accel/`](lenet5_conv_accel) | Convolution layers accelerated (fixed-point + adder tree) | 2057 µs | 2.9× | 36% | 100% |
| [`lenet5_full_accel/`](lenet5_full_accel) | Conv **and** FC accelerated (FC unroll ×16) — **best** | **69.5 µs** | **87×** | 77% | 100% |
| [`conv5x5_unoptimised/`](conv5x5_unoptimised) | Standalone single 5×5 conv accelerator (baseline) | — | — | — | 100% |

¹ Vitis HLS C-synthesis estimate for one image @200 MHz. Accuracy = MNIST 100-image
test set, verified in both float and the `ap_fixed<20,8>` fixed-point datapath.

**Full measured results and the reasoning behind each step:**
[`lenet5_full_accel/ACCELERATION_RESULTS.md`](lenet5_full_accel/ACCELERATION_RESULTS.md)
· how-to for going further: [`lenet5_full_accel/ACCELERATION_GUIDE.md`](lenet5_full_accel/ACCELERATION_GUIDE.md).

### Headline findings
- Accelerating the **conv alone** only gave 2.9× — the real bottleneck was FC1.
- Accelerating the **FC layers** unlocked the jump to 72–87×.
- **Maxing out LUTs on the conv backfired** — a channel-parallel conv hit 109% LUT
  (won't route) for no real speedup, while feeding the FC bottleneck (unroll ×16)
  was faster *and* fit at 77%.

## Each workspace's layout
Every `lenet5_*` folder is a self-contained Vitis workspace with the same
components: `lenet5_hls` (HLS kernel, top = `cnn`), `lenet5_system` (hw_link +
package), `lenet5_host` (ARM/XRT: MNIST accuracy + custom-image inference), plus
`hls/` (sources + trained weights + MNIST data) and `sample_images/`. Only the
`hls/src/core/` layer code differs between them.

## Quick verify on a host PC (no board, no Vitis)
Sources are plain C++ (HLS pragmas ignored by g++), so C-simulation runs anywhere:
```bash
cd lenet5_full_accel/hls && g++ -O2 -w -std=c++14 \
    -Iinclude/core -Iinclude/test -Iinclude/test/oracle \
    src/core/*.cpp src/test/*.cpp -o csim && ./csim 0 0 100   # -> 100% accuracy
```

## Building / deploying (teammates)
Each `lenet5_*` is opened as its own Vitis workspace. Two machine-specific things
to set on your own machine:
1. **Platform** — the components reference `kv260_custom` by name. Launch Vitis with
   `PLATFORM_REPO_PATHS=<path to your Lab1 kv260_custom/export>` (or set the platform
   in each component's settings editor).
2. **Sysroot** — edit the `sysroot` path in each `lenet5_host/vitis-comp.json` to
   your own Vitis sysroot.

Then C-Synthesis → build `lenet5_system` (→ `.bin`) and `lenet5_host`, deploy via
`xmutil` (see the AMD KV260 flow), and run `./lenet5_host -x <app>.bin`.

## Other files
- `docs/COMP4601 Project Slides.pdf` — project brief.
- `LICENSE.txt` — MIT; the LeNet core is ported from Universitat Politècnica de
  València (attribution in each `lenet5_*/README.md`).
- `UPDATES.md` — log of scope/direction changes.
