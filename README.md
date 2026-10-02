# Four-Way Parallel MDST: GPU Implementation and Scalability Analysis

This repository contains the source code associated with the research paper:

> **A Novel Low-Complexity Four-Way Parallel MDST Algorithm for GPU and Multicore CPU Architectures**  
> Doru Florin Chiper, Dan Marius Dobrea  
> Submitted to [*Eng* (MDPI)](https://www.mdpi.com/journal/eng)

The code is released to improve the reproducibility of the GPU experiments reported in the paper.

---

## Overview

The repository provides the CUDA implementations and benchmarking tools used in the experimental evaluation of the proposed parallel algorithm for the one-dimensional **Modified Discrete Sine Transform (MDST)**:

1. **Single-transform implementation:** the proposed MDST algorithm, whose four independent computational sections run in parallel on the GPU.
2. **Batched implementation:** many independent MDST transforms processed simultaneously, used to investigate GPU scalability.
3. **Profiling workflow:** Nsight Compute scripts that measure GPU utilization (achieved occupancy, SM activity and SM throughput) for each batch size.

## Repository structure

| File | Description |
|---|---|
| [`main_MDST_paralel.cu`](main_MDST_paralel.cu) | Single-transform implementation (four-section kernel, `<<<4, 1>>>`) |
| [`mdst_batched_agx_orin.cu`](mdst_batched_agx_orin.cu) | Batched implementation and scalability benchmark |
| [`build_agx_orin.sh`](build_agx_orin.sh) | Build script for the batched implementation (Jetson AGX Orin) |
| [`profile_ncu_agx_orin.sh`](profile_ncu_agx_orin.sh) | Timing benchmark and Nsight Compute profiling for all batch sizes, with merged results |

## Requirements

- NVIDIA GPU with CUDA support (the scalability experiment targets **NVIDIA Jetson AGX Orin**)
- CUDA Toolkit with the `nvcc` compiler; the batched implementation requires CUDA 11.0 or newer (`-std=c++17`)
- NVIDIA Nsight Compute CLI (`ncu`), for profiling only
- Bash and Python 3 (standard library only), for the profiling script only

> [!TIP]
> On Jetson devices, the CUDA tools are installed in `/usr/local/cuda/bin`, which is not always on the `PATH`. If `nvcc` is not found, run `export PATH=/usr/local/cuda/bin:$PATH`.

## Quick start (Jetson AGX Orin)

```bash
git clone https://github.com/dmdobrea/MDST-with-4-paralel-sections.git
cd MDST-with-4-paralel-sections
chmod +x build_agx_orin.sh profile_ncu_agx_orin.sh

# 1. Single-transform implementation
nvcc main_MDST_paralel.cu -o main_MDST_paralel -arch=sm_87
./main_MDST_paralel            # prints the elapsed time; press Enter to exit

# 2. Batched implementation and complete scalability experiment
./build_agx_orin.sh
./profile_ncu_agx_orin.sh
```

---

## 1. Single-transform implementation

### 1.1 Transform definition

For an input sequence $x(n)$, $n = 0, 1, \dots, N-1$, the MDST computes $M = N/2$ coefficients:

```math
Y(k) = \sum_{n=0}^{N-1} x(n)\,\sin\!\left[\frac{\pi}{2N}\left(2n + 1 + \frac{N}{2}\right)(2k + 1)\right], \qquad k = 0, 1, \dots, M - 1
```

Both implementations use an input length of $N = 34$ samples, producing $M = 17$ output coefficients. In the source code, these sizes are named `N1` (= 34) and `N` (= 17), respectively.

### 1.2 Four-section decomposition and thread mapping

[`main_MDST_paralel.cu`](main_MDST_paralel.cu) contains the CUDA implementation of the proposed algorithm. The proposed reformulation decomposes the MDST computation into four independent computational sections, each containing a subset of the mathematical operations of the algorithm. The sections read only the input data and write disjoint output elements, so they run in parallel without any synchronization.

The kernel is launched as `MDST4<<<4, 1>>>`, i.e., four blocks with one thread each, and each thread computes one complete section:

| Block (`blockIdx.x`) | Section | Input sequence | Partial results |
|:---:|:---:|:---:|---|
| 0 | 1 | `xa` | `Ysb[0]`, `Ysb[1]`, `Ysb[3]`, `Ysb[7]` |
| 1 | 2 | `xa` | `Ysb[2]`, `Ysb[4]`, `Ysb[5]`, `Ysb[6]` |
| 2 | 3 | `xb` | `Ycb[0]`, `Ycb[1]`, `Ycb[3]`, `Ycb[7]` |
| 3 | 4 | `xb` | `Ycb[2]`, `Ycb[4]`, `Ycb[5]`, `Ycb[6]` |

One transform is computed in three steps:

1. **Host pre-processing:** the input samples are multiplied by precomputed sine constants and combined into the two 17-sample sequences `xa` and `xb`, which are copied to the GPU.
2. **GPU computation:** the four sections run in parallel and produce the partial results `Ysb` and `Ycb` (8 values each).
3. **Host post-processing:** the 17 MDST coefficients are obtained from `Ysb`, `Ycb` and `xa` through a running sum.

This implementation evaluates the computational benefits introduced by the algorithmic reformulation and provides the baseline results reported in the paper for the proposed GPU implementation. The elapsed time of the complete pipeline (steps 1–3, including host–device transfers) is measured with CUDA events and printed at the end of the run.

### 1.3 Build and run

```bash
nvcc main_MDST_paralel.cu -o main_MDST_paralel -arch=sm_XX
./main_MDST_paralel
```

Replace `sm_XX` with the compute capability of the target GPU:

| Architecture | Example devices | Flag |
|---|---|---|
| Volta | Jetson AGX Xavier, Jetson Xavier NX | `-arch=sm_72` |
| Ampere | Jetson AGX Orin, Orin NX, Orin Nano | `-arch=sm_87` |
| Ada Lovelace | GeForce RTX 40 series | `-arch=sm_89` |

The compute capability of other GPUs is listed at <https://developer.nvidia.com/cuda-gpus>. At each run, the program processes a new random 34-sample input, prints the elapsed time (`Elapsed time: … ms`) and waits for a key press before exiting.

---

## 2. GPU scalability through batched MDST processing

### 2.1 Motivation

Although the four-section decomposition exposes algorithmic parallelism, a single $N = 34$ MDST transform provides a limited workload for modern, massively parallel GPUs. To investigate GPU scalability under realistic high-throughput conditions, a second implementation processes multiple independent MDST transforms simultaneously. This corresponds to practical scenarios such as:

- continuous time-series processing;
- overlapping signal windows;
- multi-channel processing;
- simultaneous independent transforms.

The experiment evaluates how **inter-transform parallelism** (many transforms processed at once) complements the **intra-transform parallelism** of the four-section decomposition, and characterizes the transition from latency-oriented single-transform execution towards throughput-oriented GPU execution.

### 2.2 Batched kernel design

[`mdst_batched_agx_orin.cu`](mdst_batched_agx_orin.cu) processes a batch of $B$ independent transforms with the kernel `MDST4_batched`:

- **Thread mapping:** each block contains 128 threads (4 warps). Warp $w$ computes section $w + 1$, and each of its 32 lanes handles a different transform. A block therefore processes 32 transforms, and the grid contains $\lceil B/32 \rceil$ blocks.
- **No intra-warp divergence:** the section index is derived from the warp index, so all threads of a warp follow the same branch of the section `switch`.
- **Coalesced memory accesses:** inputs and outputs are stored transposed (`[coefficient][transform]`), so adjacent lanes access consecutive memory addresses.
- **Register-level computation:** the prefix recurrence is evaluated on the fly, keeping only the eight pair sums required by the sections; the kernel uses no shared memory.
- **Test signal:** the batch is formed by $B$ windows of 34 samples with 50% overlap (hop of 17 samples), extracted from a pseudo-random signal generated with a fixed seed. The input data are therefore identical across runs.

The batch size $B$ controls the amount of available parallel work. The benchmark evaluates the following batch sizes:

`1, 32, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384`

For each batch size, two execution times are measured:

- **Kernel time:** measured with CUDA events over `--repeats` consecutive kernel launches, after `--warmup` warm-up launches.
- **End-to-end time:** host-to-device transfer, kernel execution, device-to-host transfer and host-side reconstruction of all $B$ transforms, averaged over `--e2e-repeats` iterations. Memory allocation, batch generation (including the host pre-processing that forms `xa` and `xb`) and CUDA context creation are excluded.

### 2.3 Build

```bash
./build_agx_orin.sh                                 # mdst_batched_agx_orin.cu -> mdst_batched_agx_orin
./build_agx_orin.sh <source_file.cu> <executable>   # custom source file / executable name
```

The script compiles for Jetson AGX Orin (Ampere, compute capability 8.7) with:

```bash
nvcc -O3 -std=c++17 -arch=sm_87 -lineinfo -Xptxas=-v mdst_batched_agx_orin.cu -o mdst_batched_agx_orin
```

- `-lineinfo` allows Nsight Compute to associate metrics with source lines;
- `-Xptxas=-v` reports the resources used by the kernel (registers, shared memory), saved in `ptxas_report.txt`.

All compiler messages, including errors, are redirected to `ptxas_report.txt`; check this file if the build fails. To build for a different GPU, change the `-arch` value in the script.

### 2.4 Run the benchmark

```bash
./mdst_batched_agx_orin --all --csv mdst_batch_timing.csv   # all predefined batch sizes
./mdst_batched_agx_orin --b 4096                            # a single batch size
./mdst_batched_agx_orin --b 4096 --verify                   # ... with numerical verification
```

| Option | Description | Default |
|---|---|---|
| `--all` | Run all predefined batch sizes (default when the program is started without arguments) | – |
| `--b B` | Run a single batch size `B` | – |
| `--repeats N` | Number of timed kernel launches | `1000` |
| `--warmup N` | Number of warm-up launches before timing | `20` |
| `--e2e-repeats N` | Number of end-to-end timing iterations | `20` |
| `--csv FILE` | Output CSV file | `mdst_batch_timing.csv` |
| `--verify` | Compare the first transform of the batch with the direct MDST definition | off |
| `--profile-only` | Single kernel launch without timing, used by Nsight Compute (requires `--b`) | off |
| `-h`, `--help` | Show usage information | – |

### 2.5 Timing results

The results are printed to the terminal and written to the CSV file (`mdst_batch_timing.csv` by default):

| Column | Description |
|---|---|
| `B` | Batch size (number of transforms) |
| `blocks`, `threads_per_block` | Launch configuration |
| `kernel_ms_per_batch` | Average kernel time per batch (ms) |
| `kernel_us_per_transform` | Average kernel time per transform (µs) |
| `transforms_per_s` | Kernel throughput (transforms/s) |
| `e2e_ms_per_batch` | End-to-end time per batch (ms) |
| `e2e_us_per_transform` | End-to-end time per transform (µs) |
| `theoretical_occupancy_pct` | Theoretical occupancy from the CUDA occupancy API (%) |
| `active_blocks_per_sm` | Maximum number of resident blocks per SM |
| `registers_per_thread` | Registers used per thread |
| `static_shared_bytes` | Static shared memory per block (bytes) |
| `verify_max_abs_error` | Maximum absolute error vs. the direct MDST (`NA` without `--verify`) |

---

## 3. NVIDIA Nsight Compute profiling

[`profile_ncu_agx_orin.sh`](profile_ncu_agx_orin.sh) automates the complete scalability experiment:

1. runs the timing benchmark for all batch sizes;
2. profiles each batch size with Nsight Compute, using a dedicated run with a single kernel launch (`--profile-only`);
3. merges the timing results and the GPU utilization metrics into a single CSV file.

### 3.1 Collected metrics

| Metric | Nsight Compute metric | CSV column |
|---|---|---|
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | `achieved_occupancy_pct` |
| SM activity | `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` | `sm_active_pct` |
| SM throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | `sm_throughput_pct` |

*Achieved occupancy* is the average number of active warps per active cycle, relative to the maximum supported by an SM; *SM activity* is the percentage of elapsed cycles in which the SMs had at least one active warp; *SM throughput* corresponds to the *Compute (SM) Throughput* reported in the Speed of Light section of Nsight Compute.

### 3.2 Running the complete scalability experiment

After building the batched implementation (Section 2.3):

```bash
ncu --version                # check that the Nsight Compute CLI is available
./profile_ncu_agx_orin.sh
```

Optional arguments override the executable and the output file names listed in Section 3.3:

```bash
./profile_ncu_agx_orin.sh [EXECUTABLE] [TIMING_CSV] [NCU_CSV] [MERGED_CSV]
```

> [!NOTE]
> If access to the GPU performance counters is restricted (Nsight Compute reports `ERR_NVGPUCTRPERM`), run the script with `USE_SUDO=1`, which executes only `ncu` with administrator rights:
>
> ```bash
> USE_SUDO=1 ./profile_ncu_agx_orin.sh
> ```
>
> See NVIDIA's [ERR_NVGPUCTRPERM page](https://developer.nvidia.com/ERR_NVGPUCTRPERM) for details.

### 3.3 Output files

| File | Content |
|---|---|
| `mdst_batch_timing.csv` | Timing results for all batch sizes (columns listed in Section 2.5) |
| `mdst_ncu_metrics.csv` | Nsight Compute metrics for all batch sizes |
| `mdst_agx_orin_results.csv` | **Final merged table:** timing columns + `achieved_occupancy_pct`, `sm_active_pct`, `sm_throughput_pct` |
| `ncu_raw/ncu_B<B>.csv` | Raw Nsight Compute output for each batch size |

---

## Reproducibility

The GPU experiments reported in the paper can be reproduced with the source code and scripts provided in this repository. The recommended platform for the scalability experiment is **NVIDIA Jetson AGX Orin** with the CUDA Toolkit and NVIDIA Nsight Compute.

- The batched benchmark uses a fixed random seed, so all runs process identical input data.
- On Jetson devices, execution times depend on the active power mode (`nvpmodel`) and clock configuration (`jetson_clocks`); use the same settings when comparing results.

## Citation

If you use this code, please cite the paper (the reference will be completed after publication):

> Chiper, D.F.; Dobrea, D.M. A Novel Low-Complexity Four-Way Parallel MDST Algorithm for GPU and Multicore CPU Architectures. *Eng* **[Year]**, *[Volume]*, [Article number]. DOI: [DOI]

```bibtex
@article{Chiper_MDST_Eng,
  author  = {Chiper, Doru Florin and Dobrea, Dan Marius},
  title   = {A Novel Low-Complexity Four-Way Parallel {MDST} Algorithm for {GPU} and Multicore {CPU} Architectures},
  journal = {Eng},
  year    = {[Year]},
  volume  = {[Volume]},
  pages   = {[Article number]},
  doi     = {[DOI]}
}
```

## License

[To be completed]
