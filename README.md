# Parallel MDST GPU Implementation and Scalability Analysis

This repository contains the source code associated with the research paper:

**"A Novel Low-Complexity Four-Way Parallel MDST Algorithm for GPU and Multicore CPU Architectures"**

submitted to:

**Eng (MDPI)**  
Journal link: https://www.mdpi.com/journal/eng

Authors:
- Doru Florin Chiper, Dan Marius Dobrea 

---

## Overview

This repository provides the CUDA implementations and benchmarking tools used in the experimental evaluation of the proposed parallel implementation of the one-dimensional Modified Discrete Sine Transform (MDST).

The code supports:

1. The main parallel MDST implementation based on the proposed four-section decomposition.
2. A batched MDST implementation designed to investigate GPU scalability when multiple independent transforms are processed simultaneously.
3. Nsight Compute profiling scripts used to evaluate GPU utilization metrics, including achieved occupancy, SM activity, and SM throughput.

The objective of releasing this repository is to improve reproducibility of the experimental results presented in the paper.

---

# 1. Main Parallel MDST Implementation

## File Description

`main_MDST_paralel.cu` contains the main CUDA implementation of the proposed MDST algorithm.

The proposed formulation decomposes the original MDST computation into four independent computational sections. Each section contains a subset of the mathematical operations obtained from the proposed reformulation, enabling parallel execution.

For a single MDST transform:

- Input length:

\[
N = 34
\]

- Output coefficients:

\[
M=N/2=17
\]

The CUDA implementation maps the four independent computational sections to four CUDA execution threads:

CUDA kernel configuration:
<<<4,1>>>
Block 0 -> Section 1
Block 1 -> Section 2
Block 2 -> Section 3
Block 3 -> Section 4


Each CUDA thread computes one complete section of the proposed decomposition.

The implementation is intended to evaluate the computational benefits introduced by the algorithmic reformulation and provides the baseline results reported in the paper for the proposed GPU implementation.

---

# Compilation

The implementation requires:

- NVIDIA GPU
- CUDA Toolkit
- NVIDIA CUDA compiler (`nvcc`)

Example compilation:

```bash
nvcc main_MDST_paralel.cu -o main_MDST_paralel -arch=sm_xx
'''
where:
- sm_72 corresponds to NVIDIA Volta GPUs
- sm_87 corresponds to NVIDIA Ampere GPUs (e.g., Jetson AGX Orin)
- sm_89 corresponds to NVIDIA Ada Lovelace GPUs
Example for Jetson AGX Orin:

nvcc main_MDST_paralel.cu -o main_MDST_paralel -arch=sm_87

Run:

./main_MDST_paralel

# 2. GPU Parallelism Scalability through Batched MDST Processing
## Motivation
Although the proposed four-section decomposition exposes algorithmic parallelism, a single \(N=34\) MDST transform provides limited workload for modern massively parallel GPUs.
To investigate GPU scalability under realistic high-throughput conditions, a second implementation was developed in which multiple independent MDST transforms are processed simultaneously.
This corresponds to practical scenarios such as:
- continuous time-series processing;
- overlapping signal windows;
- multi-channel processing;
- simultaneous independent transforms.
The objective of this experiment is to evaluate how inter-transform parallelism complements the four-section intra-transform parallelism.
## Batched Implementation Files
The scalability analysis uses the following files:
mdst_batched_agx_orin.cu
build_agx_orin.sh
profile_ncu_agx_orin.sh

## 2.1 Batched MDST CUDA Implementation
File
mdst_batched_agx_orin.cu

Description
This program implements batched MDST processing.
Instead of executing a single transform, the program processes:
B independent MDST transforms

where:
B =
1,
32,
128,
256,
512,
1024,
2048,
3072,
4096,
6144,
8192,
12288,
16384

The batch size \(B\) controls the amount of available parallel work.
The experiment evaluates the transition from:
- latency-oriented single-transform execution;
towards:
- throughput-oriented GPU execution.
The collected metrics include:
- execution time per transform;
- transforms per second;
- achieved occupancy;
- SM activity;
- SM throughput.
Compilation for Jetson AGX Orin
File
build_agx_orin.sh

This script automatically compiles the batched implementation for NVIDIA Jetson AGX Orin.
The compilation configuration is:
nvcc -O3 -std=c++17 \
    -arch=sm_87 \
    -lineinfo \
    -Xptxas=-v

The script also generates:
ptxas_report.txt

containing CUDA compiler resource information, including register and shared-memory usage.
Usage:
./build_agx_orin.sh

or:
./build_agx_orin.sh source_file.cu executable_name

Running the Batched Benchmark
Example:
./mdst_batched_agx_orin --all --csv mdst_batch_timing.csv

The program evaluates all predefined batch sizes and generates:
mdst_batch_timing.csv

containing timing and throughput results.
A single batch size can be evaluated using:
./mdst_batched_agx_orin --b 4096

3. NVIDIA Nsight Compute Profiling
File
profile_ncu_agx_orin.sh

This script automatically profiles each batch size using NVIDIA Nsight Compute.
The collected GPU utilization metrics are:
- achieved occupancy:
sm__warps_active.avg.pct_of_peak_sustained_active

- SM activity:
sm__cycles_active.avg.pct_of_peak_sustained_elapsed

- SM throughput:
sm__throughput.avg.pct_of_peak_sustained_elapsed

Nsight Compute Requirements
The script requires:
- NVIDIA Nsight Compute CLI (ncu)
- CUDA profiling permissions
Check availability:
ncu --version

If profiling permissions are restricted:
sudo ./profile_ncu_agx_orin.sh

Running the Complete Scalability Experiment
After compilation:
./profile_ncu_agx_orin.sh

The script performs:
1. Timing benchmark for all batch sizes.
2. Nsight Compute profiling for each batch size.
3. Automatic merging of timing and GPU utilization metrics.
Generated files:
mdst_batch_timing.csv
mdst_ncu_metrics.csv
mdst_agx_orin_results.csv

Raw Nsight Compute outputs:
ncu_raw/

The final merged CSV contains:
- batch size;
- execution time;
- throughput;
- achieved occupancy;
- SM activity;
- SM throughput.
Reproducibility
The experiments reported in the paper can be reproduced using the provided source code and scripts.
Recommended platform for the scalability experiment:
- NVIDIA Jetson AGX Orin
- CUDA Toolkit
- NVIDIA Nsight Compute
The repository provides:
- CUDA source files;
- compilation scripts;
- profiling scripts;
- benchmark generation workflow.
Citation
If you use this code, please cite:
[To be completed after publication]

Author(s),
"Title of the paper",
Eng, MDPI, Year.
DOI: [DOI]

License
[To be completed]
