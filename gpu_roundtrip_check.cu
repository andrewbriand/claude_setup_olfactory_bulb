// Platform check before profiling the CoreNEURON solver: how much does one GPU round trip
// cost here, and does it grow with managed (unified) memory?
//
// CoreNEURON allocates part of its GPU data with cudaMallocManaged (allocate_unified()).
// Without concurrent managed access -- e.g. WSL2, where concurrentManagedAccess=0 -- every
// kernel launch and sync pays for *all* managed memory the process holds (~2.8 us per MB on
// an RTX 4090 under WSL2), which then dominates solver timings. On native Linux with a
// recent GPU, concurrentManagedAccess=1 and the cost should stay flat.
//
//   source env.sh
//   nvcc -O2 -arch=sm_${CUDA_ARCH} gpu_roundtrip_check.cu -o runs/gpu_roundtrip_check
//   runs/gpu_roundtrip_check
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void touch(char* p) {
    p[threadIdx.x] += 1;
}

static double launch_sync_us(char* p, cudaStream_t s, int n) {
    auto a = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        touch<<<1, 32, 0, s>>>(p);
        cudaStreamSynchronize(s);
    }
    auto b = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(b - a).count() / n;
}

int main() {
    int dev = 0, cma = 0, pma = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, dev);
    cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, dev);
    printf("%s: concurrentManagedAccess=%d pageableMemoryAccess=%d\n", prop.name, cma, pma);

    cudaStream_t s;
    cudaStreamCreate(&s);
    char* d;
    cudaMalloc(&d, 4096);
    for (int i = 0; i < 100; ++i) {  // warm-up
        touch<<<1, 32, 0, s>>>(d);
        cudaStreamSynchronize(s);
    }
    printf("launch + sync, device memory only:        %8.1f us\n", launch_sync_us(d, s, 2000));

    for (size_t mb : {100, 1000}) {
        char* m;
        if (cudaMallocManaged(&m, mb << 20) != cudaSuccess) {
            printf("cudaMallocManaged(%zu MB) failed\n", mb);
            return 1;
        }
        for (size_t i = 0; i < (mb << 20); i += 4096)
            m[i] = 0;  // first touch on the host
        launch_sync_us(m, s, 20);
        printf("launch + sync, holding %5zu MB managed:    %8.1f us\n", mb, launch_sync_us(m, s, 200));
        cudaFree(m);
    }
    return 0;
}
