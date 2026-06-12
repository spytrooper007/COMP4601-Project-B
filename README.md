# [Hardware Acceleration of "something" on the Kria KV260]

## 1. Team Information

**Team Name: IDK yet tbh**

| Member | zID | Role |
|---|---|---|
| Gilbert Tong | | |
| Shuruthy Dhushiyandan | |  |
| Iris Martin | |  |
| Kaira Dumasia | |  |

## 2. Project Overview / Abstract

- What problem are we solving?
- Why does FPGA acceleration make sense for this problem?
- What is the expected outcome?

## 3. Motivation

- Why does this algorithm/application matter?
- Why are CPUs insufficient (throughput, latency, power)?
- Why is FPGA acceleration promising (parallelism, pipelining, energy efficiency)?

## 4. Objectives

### Primary Goals
- [ ] Run software baseline on KV260
- [ ] Profile and identify hotspots
- [ ] Accelerate kernel using HLS
- [ ] Achieve ≥2× speedup
- [ ] Measure power/energy improvement

### Stretch Goals
- [ ] DMA optimisation
- [ ] Further resource tuning
- [ ] Additional features / modes

## 5. Background Research

- Algorithm explanation
- Existing software/hardware implementations
- Prior acceleration approaches
- Relevant papers, specs, tools

*(Add algorithm pipeline / dataflow diagrams here)*

## 6. System Architecture

### Hardware Architecture (PL)
- Block diagram (PS vs PL partition)
- IP core(s) design

### Software Architecture (PS)
- Application structure on ARM cores

### Communication Strategy
- AXI-Lite (control), AXI-Stream (streaming data), DMA (bulk transfers)
- Justification for chosen approach

## 7. Development Plan / Timeline

| Week | Goal |
|---|---|
| 3 | Team formation + project proposal |
| 4 | Software baseline implementation |
| 5 | Profiling + project plan |
| 6 | HLS baseline |
| 7 | HLS optimisation |
| 8 | PS–PL integration |
| 9 | Benchmarking |
| 10 | Demo, presentation & report |

## 8. Software Baseline

- Description of original (unaccelerated) implementation
- Correctness testing methodology
- Profiling results / hotspot identification

**Metrics:** runtime, CPU usage, energy

## 9. Profiling Results

- Timing breakdown / flame graph
- Identified bottlenecks
- Justification: what should be accelerated and why?

## 10. Hardware Acceleration Design

### Baseline HLS
- Direct C-to-HLS port, correctness focus

### Optimisations
- Pipelining
- Loop unrolling
- Memory optimisation (burst, local buffers, ping-pong)
- Parallelisation / dataflow

### Interfaces
- AXI interface definitions, ports, protocols used

## 11. Implementation Details

- Tool versions (Vivado, Vitis HLS, PetaLinux, etc.)
- Build flow / synthesis steps
- Repository structure:

```
/src        - source code (PS + HLS kernels)
/hw         - block design, constraints, bitstreams
/sw         - host application
/scripts    - build / test automation
/docs       - reports, diagrams
/results    - benchmark data, logs
```

## 12. Testing & Validation

- Test methodology
- Verification approach (HLS C-simulation, co-simulation, on-board)
- Correctness comparison vs. software reference
- Unit tests and edge cases covered

## 13. Performance Evaluation

| Metric | Software Baseline | Hardware Accelerated | Speedup |
|---|---|---|---|
| Latency | | | |
| Throughput | | | |
| Power | | | |

### Resource Utilisation

| Resource | Used | Available | % |
|---|---|---|---|
| LUT | | | |
| FF | | | |
| BRAM | | | |
| DSP | | | |

### Scalability
- How do results change with problem size?

## 14. Results & Discussion

- Did acceleration help as expected?
- What became the new bottlenecks?
- Any unexpected findings (e.g. divergence between predicted vs actual speedup)?

## 15. Challenges & Lessons Learned

- AXI/interconnect bottlenecks
- DMA complexity
- Timing closure issues
- HLS tool limitations / quirks

## 16. Future Work

- Larger datasets
- Better scheduling / pipelining
- Custom memory hierarchy
- Additional algorithm variants

## 17. Repository / Deliverables

- **GitHub repo:** [link]
- **Build instructions:** see `/docs/BUILD.md`
- **Demo video:** [link]
- **Final report:** [link]

## 18. References

*(IEEE or APA format)*

1.
2.
3.

## 19. Appendices (optional)

- Extra benchmark tables
- Full resource utilisation reports
- Code snippets
- Timing reports
