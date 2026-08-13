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
| [`lenet5_full_accel/`](lenet5_full_accel) | Conv **and** FC accelerated (FC unroll ×16) — **best / deployed** | **69.5 µs** | **87×** | 77% | 100% |
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

---

# Testing `lenet5_full_accel` yourself (step by step)

`lenet5_full_accel` is the best deployed design. There are three levels of
testing, from easiest to most complete. **Level 1 needs nothing but a C++
compiler** — start there.

## Level 1 — Verify accuracy on any PC (no board, no Vitis, ~2 min)

The HLS kernel is plain C++ (the hardware pragmas are ignored by a normal
compiler), so the whole network runs on any Linux/macOS machine with `g++`.

1. Unzip the project and open a terminal.
2. Go into the workspace's `hls` folder:
   ```bash
   cd lenet5_full_accel/hls
   ```
3. Compile the network + testbench:
   ```bash
   g++ -O2 -w -std=c++14 -Iinclude/core -Iinclude/test -Iinclude/test/oracle \
       src/core/*.cpp src/test/*.cpp -o csim
   ```
4. Classify the 100 bundled MNIST test images:
   ```bash
   ./csim 0 0 100
   ```
   The three numbers are `<mode> <from-image> <to-image>`; `0` selects the whole
   network. You should see one line per image ending in:
   ```
   The accuracy of the test_cnn operation is 100.000000
   ```

That confirms the design is functionally correct. (This is the **float** path;
the on-board hardware uses `ap_fixed<20,8>`, which we verified separately also
gives 100%.)

## Level 2 — Reproduce the performance numbers in Vitis (C-synthesis, no board)

This regenerates the 69.5 µs latency and 77% LUT figures. Needs **AMD Vitis
2025.x** (Unified IDE) installed.

1. Make the platform discoverable. In a terminal, source your Vitis settings, then:
   ```bash
   export PLATFORM_REPO_PATHS=<path to your kv260_custom export folder>
   vitis -w lenet5_full_accel
   ```
   (Or launch Vitis normally and set the platform in each component's settings.)
2. In the **Vitis Components** panel, expand the **`lenet5_hls`** component.
3. Run **C Simulation** (Flow ▸ C Simulation). It builds and classifies image 0.
   To sweep all 100, set the testbench arguments to `0 0 100` in the run settings.
4. Run **C Synthesis** (Flow ▸ C Synthesis). When it finishes, open the report:
   `lenet5_hls/cnn/hls/syn/report/csynth.rpt`. Check:
   - **Total latency** ≈ 13,893 cycles (× 5 ns = **69.5 µs** @200 MHz)
   - **Utilization**: LUT ≈ 77%, DSP ≈ 22%, BRAM ≈ 58%
   - Per-loop **II = 1** on the conv and FC pipelines, timing met (positive slack).

## Level 3 — Run on the actual KV260 board

Needs the board, Vitis, and the `kv260_custom` platform you built in Lab 1.

1. **Set two machine-specific paths** in the workspace (they point at *this*
   author's machine):
   - Platform: launch Vitis with `PLATFORM_REPO_PATHS` set (as in Level 2), or set
     the platform in the `lenet5_system` / `lenet5_host` settings editors.
   - Sysroot: edit the `sysroot` line in
     `lenet5_full_accel/lenet5_host/vitis-comp.json` to your own Vitis sysroot.
2. **Build the bitstream and host** (target `hw`):
   - Build **`lenet5_system`** → produces
     `lenet5_system/build/hw/hw_link/binary_container_1.xclbin`
   - Build **`lenet5_host`** → produces `lenet5_host/build/hw/lenet5_host`
3. **Copy to the board** (replace the IP with your board's). Rename the xclbin to
   `lenet5.bin`:
   ```bash
   BOARD=petalinux@<board-ip>
   scp lenet5_full_accel/lenet5_system/build/hw/hw_link/binary_container_1.xclbin $BOARD:/home/petalinux/lenet5.bin
   scp lenet5_full_accel/lenet5_host/build/hw/lenet5_host                         $BOARD:/home/petalinux/
   scp <your kv260_custom export>/sw/boot/pl.dtbo                                 $BOARD:/home/petalinux/
   scp <a generic XRT_FLAT shell.json>                                            $BOARD:/home/petalinux/
   scp -r lenet5_full_accel/sample_images                                         $BOARD:/home/petalinux/
   ```
4. **Install and load the app** (on the board, over SSH):
   ```bash
   sudo mkdir -p /lib/firmware/xilinx/lenet5
   sudo cp ~/lenet5.bin ~/pl.dtbo ~/shell.json /lib/firmware/xilinx/lenet5/
   sudo xmutil unloadapp
   sudo xmutil loadapp lenet5
   ```
   Expect `lenet5: Loaded with slot_handle 0` (overlay `memory` warnings are
   harmless). If a slot wedges, `sudo reboot` and re-load.
5. **Run it:**
   ```bash
   chmod +x ~/lenet5_host && ./lenet5_host -x lenet5.bin
   ```
   Prints per-image predictions, overall accuracy, and the FPGA-vs-ARM timing.
6. **Classify your own digit** (28×28, 784 numbers, row-major):
   ```bash
   ./lenet5_host -x lenet5.bin -i sample_images/digit7_raw.txt          # raw 0..255 (default)
   ./lenet5_host -x lenet5.bin -i sample_images/digit7_std.txt -f std   # already normalized
   ```

---

## Each workspace's layout
Every `lenet5_*` folder is a self-contained Vitis workspace with the same
components: `lenet5_hls` (HLS kernel, top = `cnn`), `lenet5_system` (hw_link +
package), `lenet5_host` (ARM/XRT: MNIST accuracy + custom-image inference), plus
`hls/` (sources + trained weights + MNIST data) and `sample_images/`. Only the
`hls/src/core/` layer code differs between them — so you can test any variant with
the same Level 1 command, just changing the folder name.

## Other files
- `docs/COMP4601 Project Slides.pdf` — project brief.
- `LICENSE.txt` — MIT; the LeNet core is ported from Universitat Politècnica de
  València (attribution in each `lenet5_*/README.md`).
- `UPDATES.md` — log of scope/direction changes.
