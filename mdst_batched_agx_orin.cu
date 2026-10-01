#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                     cudaGetErrorString(_e));                                     \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

constexpr int N1       = 34;          // input samples / MDST window
constexpr int N        = 17;          // N1/2 output coefficients

constexpr int HALF_OUT        = 8;  // (N-1)/2
constexpr int HOP             = 17; // 50% overlap: N1/2
constexpr int WARPS_PER_BLOCK = 4;  // one warp per MDST section
constexpr int WARP_SIZE_      = 32;

constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE_; // 128
constexpr double PI_ = 3.141592653589793238462643383279502884;

// The input arrays are stored transposed as [coefficient][transform].
// This makes accesses by adjacent lanes of a warp contiguous/coalesced.

__device__ __forceinline__ void compute_pair_features(
    const double* __restrict__ a,
    int B,
    int t,
    double& vA0, double& vA1, double& vA2, double& vA3,
    double& vB0, double& vB1, double& vB2, double& vB3,
    double& x0)
{
    // Recurrence used by the original implementation:
    // r[16] = a[16], r[i] = a[i] - r[i+1].
    // Instead of storing all 17 values, form the eight pair sums on the fly.
    double r = a[16 * B + t];
    vB3 = r;  // r[16], later + r[1]

    r = a[15 * B + t] - r; vB2 = r;  // r[15], later + r[2]
    r = a[14 * B + t] - r; vA0 = r;  // r[14], later + r[3]
    r = a[13 * B + t] - r; vB1 = r;  // r[13], later + r[4]
    r = a[12 * B + t] - r; vA2 = r;  // r[12], later + r[5]
    r = a[11 * B + t] - r; vA3 = r;  // r[11], later + r[6]
    r = a[10 * B + t] - r; vA1 = r;  // r[10], later + r[7]
    r = a[ 9 * B + t] - r; vB0 = r;  // r[9],  later + r[8]

    r = a[ 8 * B + t] - r; vB0 += r;
    r = a[ 7 * B + t] - r; vA1 += r;
    r = a[ 6 * B + t] - r; vA3 += r;
    r = a[ 5 * B + t] - r; vA2 += r;
    r = a[ 4 * B + t] - r; vB1 += r;
    r = a[ 3 * B + t] - r; vA0 += r;
    r = a[ 2 * B + t] - r; vB2 += r;
    r = a[ 1 * B + t] - r; vB3 += r;
    r = a[ 0 * B + t] - r; x0 = r;
}

// Batched mapping:
//   warp 0 -> section 0 for 32 transforms
//   warp 1 -> section 1 for 32 transforms
//   warp 2 -> section 2 for 32 transforms
//   warp 3 -> section 3 for 32 transforms
// Therefore, each 128-thread block processes up to 32 MDST transforms.
__global__ void MDST4_batched(
    const double* __restrict__ xa,
    const double* __restrict__ xb,
    double* __restrict__ Ysb,
    double* __restrict__ Ycb,
    int B)
{
    const int section = threadIdx.x >> 5;         // 0..3
    const int lane    = threadIdx.x & 31;         // 0..31
    const int t       = blockIdx.x * 32 + lane;   // transform index

    if (t >= B) return;

    double vA0, vA1, vA2, vA3;
    double vB0, vB1, vB2, vB3;
    double base0;

    if (section < 2) {
        compute_pair_features(xa, B, t,
                              vA0, vA1, vA2, vA3,
                              vB0, vB1, vB2, vB3, base0);
    } else {
        compute_pair_features(xb, B, t,
                              vA0, vA1, vA2, vA3,
                              vB0, vB1, vB2, vB3, base0);
    }

    double pA0, pA1, pA2, pA3, pA4;
    double pB0, pB1, pB2, pB3, pB4;
    double q0, q1, q2, q3;

    switch (section) {
        case 0:
            pA0 = (vA0 + vA1 + vA2 + vA3)                       * (-0.6403882032);
            pA1 = (vA0 - vA1 + vA2 - vA3)                       * (-0.8124635689);
            pA2 = (vA0       - vA2)                             * ( 0.1237912497);
            pA3 = (vA0 + vA1 - vA2 - vA3 + vA1 - vA3)           * (-0.5956100962);
            pA4 = (      vA1       - vA3)                       * (-0.7194013458);

            pB0 = (vB0 + vB1 + vB2 + vB3)                       * ( 0.3903882032);
            pB1 = (vB0 - vB1 + vB2 - vB3)                       * ( 0.6343523857);
            pB2 = (vB0       - vB2)                             * ( 0.4201019350);
            pB3 = (vB0 + vB1 - vB2 - vB3 + vB1 - vB3)           * ( 2.1420839519);
            pB4 = (      vB1       - vB3)                       * ( 1.7219820169);

            q0 = pB0 + pB1 + pB2       - pB4 + pB2 + pA0 + pA1 + pA2       - pA4 + pA2;
            q1 = pB0 - pB1 - pB2 + pB3 - pB4 - pB4 + pA0 - pA1 - pA2 + pA3 - pA4 - pA4;
            q2 = pB0 + pB1 - pB2       + pB4 - pB2 + pA0 + pA1 - pA2       + pA4 - pA2;
            q3 = pB0 - pB1 + pB2 - pB3 + pB4 + pB4 + pA0 - pA1 + pA2 - pA3 + pA4 + pA4;

            Ysb[0 * B + t] = (base0 + q3) * 0.9829730996839018;
            Ysb[1 * B + t] = (base0 + q0) * 0.9324722294043558;
            Ysb[3 * B + t] = (base0 + q1) * 0.7390089172206591;
            Ysb[7 * B + t] = (base0 + q2) * 0.0922683594633020;
            break;

        case 1:
            pA0 = (vA0 + vA1 + vA2 + vA3)                       * ( 0.3903882032);
            pA1 = (vA0 - vA1 + vA2 - vA3)                       * (-0.6343523857);
            pA2 = (vA0       - vA2)                             * ( 0.8609910085);
            pA3 = (vA0 + vA1 - vA2 - vA3 + vA1 - vA3)           * ( 0.0207871385);
            pA4 = (      vA1       - vA3)                       * (-0.8402038699);

            pB0 = (vB0 + vB1 + vB2 + vB3)                       * (-0.6403882032);
            pB1 = (vB0 - vB1 + vB2 - vB3)                       * (-0.8124635689);
            pB2 = (vB0       - vB2)                             * ( 0.1237912497);
            pB3 = (vB0 + vB1 - vB2 - vB3 + vB1 - vB3)           * (-0.5956100962);
            pB4 = (      vB1       - vB3)                       * (-0.7194013458);

            q0 = pB0 + pB1 + pB2       - pB4 + pB2 + pA0 + pA1 + pA2       - pA4 + pA2;
            q1 = pB0 - pB1 - pB2 + pB3 - pB4 - pB4 + pA0 - pA1 - pA2 + pA3 - pA4 - pA4;
            q2 = pB0 + pB1 - pB2       + pB4 - pB2 + pA0 + pA1 - pA2       + pA4 - pA2;
            q3 = pB0 - pB1 + pB2 - pB3 + pB4 + pB4 + pA0 - pA1 + pA2 - pA3 + pA4 + pA4;

            Ysb[2 * B + t] = (base0 + q2) * 0.8502171357296142;
            Ysb[4 * B + t] = (base0 + q0) * 0.6026346363792563;
            Ysb[5 * B + t] = (base0 + q3) * 0.4457383557765383;
            Ysb[6 * B + t] = (base0 + q1) * 0.2736629900720829;
            break;

        case 2:
            pA0 = (vA0 + vA1 + vA2 + vA3)                       * (-0.6403882032);
            pA1 = (vA0 - vA1 + vA2 - vA3)                       * (-0.8124635689);
            pA2 = (vA0       - vA2)                             * ( 0.1237912497);
            pA3 = (vA0 + vA1 - vA2 - vA3 + vA1 - vA3)           * (-0.5956100962);
            pA4 = (      vA1       - vA3)                       * (-0.7194013458);

            pB0 = (vB0 + vB1 + vB2 + vB3)                       * ( 0.3903882032);
            pB1 = (vB0 - vB1 + vB2 - vB3)                       * ( 0.6343523857);
            pB2 = (vB0       - vB2)                             * ( 0.4201019350);
            pB3 = (vB0 + vB1 - vB2 - vB3 + vB1 - vB3)           * ( 2.1420839519);
            pB4 = (      vB1       - vB3)                       * ( 1.7219820169);

            q0 = pB0 + pB1 + pB2       - pB4 + pB2 + pA0 + pA1 + pA2       - pA4 + pA2;
            q1 = pB0 - pB1 - pB2 + pB3 - pB4 - pB4 + pA0 - pA1 - pA2 + pA3 - pA4 - pA4;
            q2 = pB0 + pB1 - pB2       + pB4 - pB2 + pA0 + pA1 - pA2       + pA4 - pA2;
            q3 = pB0 - pB1 + pB2 - pB3 + pB4 + pB4 + pA0 - pA1 + pA2 - pA3 + pA4 + pA4;

            Ycb[0 * B + t] = (base0 + q3) * 0.9829730996839018;
            Ycb[1 * B + t] = (base0 + q0) * 0.9324722294043558;
            Ycb[3 * B + t] = (base0 + q1) * 0.7390089172206591;
            Ycb[7 * B + t] = (base0 + q2) * 0.0922683594633020;
            break;

        case 3:
            pA0 = (vA0 + vA1 + vA2 + vA3)                       * ( 0.3903882032);
            pA1 = (vA0 - vA1 + vA2 - vA3)                       * (-0.6343523857);
            pA2 = (vA0       - vA2)                             * ( 0.8609910085);
            pA3 = (vA0 + vA1 - vA2 - vA3 + vA1 - vA3)           * ( 0.0207871385);
            pA4 = (      vA1       - vA3)                       * (-0.8402038699);

            pB0 = (vB0 + vB1 + vB2 + vB3)                       * (-0.6403882032);
            pB1 = (vB0 - vB1 + vB2 - vB3)                       * (-0.8124635689);
            pB2 = (vB0       - vB2)                             * ( 0.1237912497);
            pB3 = (vB0 + vB1 - vB2 - vB3 + vB1 - vB3)           * (-0.5956100962);
            pB4 = (      vB1       - vB3)                       * (-0.7194013458);

            q0 = pB0 + pB1 + pB2       - pB4 + pB2 + pA0 + pA1 + pA2       - pA4 + pA2;
            q1 = pB0 - pB1 - pB2 + pB3 - pB4 - pB4 + pA0 - pA1 - pA2 + pA3 - pA4 - pA4;
            q2 = pB0 + pB1 - pB2       + pB4 - pB2 + pA0 + pA1 - pA2       + pA4 - pA2;
            q3 = pB0 - pB1 + pB2 - pB3 + pB4 + pB4 + pA0 - pA1 + pA2 - pA3 + pA4 + pA4;

            Ycb[2 * B + t] = (base0 + q2) * 0.8502171357296142;
            Ycb[4 * B + t] = (base0 + q0) * 0.6026346363792563;
            Ycb[5 * B + t] = (base0 + q3) * 0.4457383557765383;
            Ycb[6 * B + t] = (base0 + q1) * 0.2736629900720829;
            break;
    }
}

static const double SIN_XS[N1] = {
     0.7390089172,  0.7980172273,  0.8502171357,  0.8951632914,
     0.9324722294,  0.9618256432,  0.9829730997,  0.9957341763,
     1.0000000000,  0.9957341763,  0.9829730997,  0.9618256432,
     0.9324722294,  0.8951632914,  0.8502171357,  0.7980172273,
     0.7390089172,  0.6736956436,  0.6026346364,  0.5264321629,
     0.4457383558,  0.3612416662,  0.2736629901,  0.1837495178,
     0.0922683595,  0.0000000000, -0.0922683595, -0.1837495178,
    -0.2736629901, -0.3612416662, -0.4457383558, -0.5264321629,
    -0.6026346364, -0.6736956436
};

struct BatchData {
    std::vector<double> xa;      // [N][B]
    std::vector<double> xb;      // [N][B]
    std::vector<double> first_x; // original 34 samples for optional verification
};

BatchData build_overlapped_batch(int B, unsigned long long seed = 123456789ULL)
{
    BatchData d;
    d.xa.resize(static_cast<size_t>(N) * B);
    d.xb.resize(static_cast<size_t>(N) * B);
    d.first_x.resize(N1);

    // B windows of length 34, 50% overlap -> hop = 17.
    const size_t signal_len = static_cast<size_t>(N1) + static_cast<size_t>(B - 1) * HOP;
    std::vector<double> signal(signal_len);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (double& v : signal) v = dist(rng);

    for (int b = 0; b < B; ++b) {
        const size_t start = static_cast<size_t>(b) * HOP;
        double xs[N1];

        for (int i = 0; i < N1; ++i) {
            const double xv = signal[start + i];
            if (b == 0) d.first_x[i] = xv;
            xs[i] = xv * SIN_XS[i];
        }

        int sign = -1;
        for (int i = 0; i < N; ++i) {
            const double a = xs[i] + xs[N1 - 1 - i];
            const double diff = xs[i] - xs[N1 - 1 - i];
            const double bb = (sign < 0) ? -diff : diff;
            sign = -sign;

            // Transposed layout: coefficient-major, transform-minor.
            d.xa[static_cast<size_t>(i) * B + b] = a;
            d.xb[static_cast<size_t>(i) * B + b] = bb;
        }
    }
    return d;
}

void reconstruct_transform(int t, int B,
                           const std::vector<double>& xa,
                           const std::vector<double>& Ysb,
                           const std::vector<double>& Ycb,
                           double out[N])
{
    double T[N] = {0.0};
    int sign = -1;
    for (int k = 1; k <= (N - 1) / 2; ++k) {
        const double ta = 2.0 * Ysb[static_cast<size_t>(k - 1) * B + t];
        const double tb = 2.0 * Ycb[static_cast<size_t>(k - 1) * B + t];
        T[2 * k - 1]     = (sign < 0) ? -ta : ta;
        T[N - 2 * k - 1] = (sign < 0) ? -tb : tb;
        sign = -sign;
    }

    out[0] = 0.0;
    for (int i = 0; i < N; ++i)
        out[0] += xa[static_cast<size_t>(i) * B + t];

    for (int k = 1; k < N; ++k)
        out[k] = T[k - 1] + out[k - 1];
}

void direct_mdst(const std::vector<double>& x, double out[N])
{
    const double alpha = PI_ / (2.0 * N1);
    for (int k = 0; k < N; ++k) {
        double s = 0.0;
        for (int i = 0; i < N1; ++i) {
            const double angle = (2.0 * i + 1.0 + N1 / 2.0) * (2.0 * k + 1.0) * alpha;
            s += x[i] * std::sin(angle);
        }
        out[k] = s;
    }
}

struct Result {
    int B = 0;
    int blocks = 0;
    int block_threads = BLOCK_THREADS;
    double kernel_ms_per_batch = 0.0;
    double kernel_us_per_transform = 0.0;
    double transforms_per_s = 0.0;
    double e2e_ms_per_batch = 0.0;
    double e2e_us_per_transform = 0.0;
    double theoretical_occupancy_pct = 0.0;
    int active_blocks_per_sm = 0;
    int registers_per_thread = 0;
    size_t static_shared_bytes = 0;
    double verify_max_abs_error = -1.0;
};

Result run_one(int B, int warmup, int repeats, int e2e_repeats, bool verify, bool profile_only)
{
    Result r;
    r.B = B;
    r.blocks = (B + 31) / 32;

    BatchData h = build_overlapped_batch(B, 123456789ULL);
    std::vector<double> hYsb(static_cast<size_t>(HALF_OUT) * B, 0.0);
    std::vector<double> hYcb(static_cast<size_t>(HALF_OUT) * B, 0.0);

    double *d_xa = nullptr, *d_xb = nullptr, *d_Ysb = nullptr, *d_Ycb = nullptr;
    const size_t in_bytes  = static_cast<size_t>(N) * B * sizeof(double);
    const size_t out_bytes = static_cast<size_t>(HALF_OUT) * B * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_xa, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_xb, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_Ysb, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_Ycb, out_bytes));

    CUDA_CHECK(cudaMemcpy(d_xa, h.xa.data(), in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xb, h.xb.data(), in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Ysb, 0, out_bytes));
    CUDA_CHECK(cudaMemset(d_Ycb, 0, out_bytes));

    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, MDST4_batched));
    r.registers_per_thread = attr.numRegs;
    r.static_shared_bytes = attr.sharedSizeBytes;

    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, MDST4_batched, BLOCK_THREADS, 0));
    r.active_blocks_per_sm = active_blocks;

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    r.theoretical_occupancy_pct =
        100.0 * static_cast<double>(active_blocks * BLOCK_THREADS) /
        static_cast<double>(prop.maxThreadsPerMultiProcessor);
    r.theoretical_occupancy_pct = std::min(100.0, r.theoretical_occupancy_pct);

    if (profile_only) {
        // Exactly one target kernel launch. This makes NCU scripting unambiguous.
        MDST4_batched<<<r.blocks, BLOCK_THREADS>>>(d_xa, d_xb, d_Ysb, d_Ycb, B);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "PROFILE_ONLY B=" << B
                  << " blocks=" << r.blocks
                  << " threads/block=" << BLOCK_THREADS
                  << " theoretical_occupancy_pct=" << std::fixed << std::setprecision(2)
                  << r.theoretical_occupancy_pct
                  << " regs/thread=" << r.registers_per_thread
                  << " static_shared_bytes=" << r.static_shared_bytes << "\n";

        CUDA_CHECK(cudaFree(d_xa));
        CUDA_CHECK(cudaFree(d_xb));
        CUDA_CHECK(cudaFree(d_Ysb));
        CUDA_CHECK(cudaFree(d_Ycb));
        return r;
    }

    for (int i = 0; i < warmup; ++i)
        MDST4_batched<<<r.blocks, BLOCK_THREADS>>>(d_xa, d_xb, d_Ysb, d_Ycb, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i)
        MDST4_batched<<<r.blocks, BLOCK_THREADS>>>(d_xa, d_xb, d_Ysb, d_Ycb, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    r.kernel_ms_per_batch = static_cast<double>(elapsed_ms) / repeats;
    r.kernel_us_per_transform = r.kernel_ms_per_batch * 1000.0 / B;
    r.transforms_per_s = static_cast<double>(B) * 1000.0 / r.kernel_ms_per_batch;

    CUDA_CHECK(cudaMemcpy(hYsb.data(), d_Ysb, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hYcb.data(), d_Ycb, out_bytes, cudaMemcpyDeviceToHost));

    if (verify) {
        double y_fast[N], y_ref[N];
        reconstruct_transform(0, B, h.xa, hYsb, hYcb, y_fast);
        direct_mdst(h.first_x, y_ref);
        double maxerr = 0.0;
        for (int k = 0; k < N; ++k)
            maxerr = std::max(maxerr, std::abs(y_fast[k] - y_ref[k]));
        r.verify_max_abs_error = maxerr;
    }

    // End-to-end batch time: H2D + kernel + D2H + host reconstruction.
    // Allocation, batch generation, and CUDA-context initialization are excluded.
    // This is intentionally separate from the kernel-only throughput metric.
    double sink = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < e2e_repeats; ++rep) {
        CUDA_CHECK(cudaMemcpy(d_xa, h.xa.data(), in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xb, h.xb.data(), in_bytes, cudaMemcpyHostToDevice));

        MDST4_batched<<<r.blocks, BLOCK_THREADS>>>(d_xa, d_xb, d_Ysb, d_Ycb, B);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hYsb.data(), d_Ysb, out_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hYcb.data(), d_Ycb, out_bytes, cudaMemcpyDeviceToHost));

        // Reconstruct all transforms exactly as in the original host-side final stage.
        for (int b = 0; b < B; ++b) {
            double y[N];
            reconstruct_transform(b, B, h.xa, hYsb, hYcb, y);
            sink += y[N - 1]; // prevents dead-code elimination
        }
    }
    
    auto t1 = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> wall = t1 - t0;
    r.e2e_ms_per_batch = wall.count() / e2e_repeats;
    r.e2e_us_per_transform = r.e2e_ms_per_batch * 1000.0 / B;

    // Make sink observable without polluting normal output.
    if (sink == 1.23456789012345e300) std::cerr << sink << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_xa));
    CUDA_CHECK(cudaFree(d_xb));
    CUDA_CHECK(cudaFree(d_Ysb));
    CUDA_CHECK(cudaFree(d_Ycb));

    return r;
}

void print_header(std::ostream& os)
{
    os << "B,blocks,threads_per_block,kernel_ms_per_batch,kernel_us_per_transform,"
          "transforms_per_s,e2e_ms_per_batch,e2e_us_per_transform,"
          "theoretical_occupancy_pct,active_blocks_per_sm,registers_per_thread,"
          "static_shared_bytes,verify_max_abs_error\n";
}

void print_result(std::ostream& os, const Result& r)
{
    os << r.B << ','
       << r.blocks << ','
       << r.block_threads << ','
       << std::fixed << std::setprecision(9)
       << r.kernel_ms_per_batch << ','
       << r.kernel_us_per_transform << ','
       << std::setprecision(3) << r.transforms_per_s << ','
       << std::setprecision(9) << r.e2e_ms_per_batch << ','
       << r.e2e_us_per_transform << ','
       << std::setprecision(3) << r.theoretical_occupancy_pct << ','
       << r.active_blocks_per_sm << ','
       << r.registers_per_thread << ','
       << r.static_shared_bytes << ',';
    if (r.verify_max_abs_error >= 0.0)
        os << std::scientific << std::setprecision(6) << r.verify_max_abs_error;
    else
        os << "NA";
    os << '\n';
}

void usage(const char* p)
{
    std::cout
        << "Usage:\n"
        << "  " << p << " --all [--repeats N] [--warmup N] [--e2e-repeats N] [--csv file] [--verify]\n"
        << "  " << p << " --b B   [--repeats N] [--warmup N] [--e2e-repeats N] [--csv file] [--verify]\n"
        << "  " << p << " --b B --profile-only\n\n"
        << "Default B values for Jetson AGX Orin:\n"
        << "  1, 32, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384\n";
}

int main(int argc, char** argv)
{
    const std::vector<int> default_B = {
        1,    32,   128,  256,  512,   1024, 2048,
        3072, 4096, 6144, 8192, 12288, 16384
    };

    bool run_all = (argc == 1);
    bool profile_only = false;
    bool verify       = false;
    int single_B      = -1;
    int repeats       = 1000;
    int warmup        = 20;
    int e2e_repeats   = 20;
    std::string csv_name = "mdst_batch_timing.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--all") run_all = true;
        else if (a == "--b" && i + 1 < argc) { single_B = std::atoi(argv[++i]); run_all = false; }
        else if (a == "--repeats" && i + 1 < argc) repeats = std::max(1, std::atoi(argv[++i]));
        else if (a == "--warmup" && i + 1 < argc) warmup = std::max(0, std::atoi(argv[++i]));
        else if (a == "--e2e-repeats" && i + 1 < argc) e2e_repeats = std::max(1, std::atoi(argv[++i]));
        else if (a == "--csv" && i + 1 < argc) csv_name = argv[++i];
        else if (a == "--verify") verify = true;
        else if (a == "--profile-only") profile_only = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else {
            std::cerr << "Unknown/incomplete option: " << a << "\n";
            usage(argv[0]);
            return 2;
        }
    }

    if (!run_all && single_B <= 0) {
        std::cerr << "--b requires B > 0\n";
        return 2;
    }
    if (profile_only && run_all) {
        std::cerr << "--profile-only must be used with a single --b value.\n";
        return 2;
    }

    // Force CUDA context creation before benchmark/profiling.
    CUDA_CHECK(cudaFree(nullptr));

    if (profile_only) {
        run_one(single_B, 0, 1, 1, false, true);
        return 0;
    }

    std::vector<int> values = run_all ? default_B : std::vector<int>{single_B};

    std::ofstream csv(csv_name);
    if (!csv) {
        std::cerr << "Cannot open CSV output: " << csv_name << "\n";
        return 1;
    }
    print_header(csv);
    print_header(std::cout);

    for (int B : values) {
        Result r = run_one(B, warmup, repeats, e2e_repeats, verify, false);
        print_result(csv, r);
        print_result(std::cout, r);
        csv.flush();
    }

    std::cerr << "\nTiming results written to: " << csv_name << "\n";
    return 0;
}
