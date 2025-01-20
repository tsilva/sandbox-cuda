#include <stdio.h>
#include <chrono>

// CUDA kernel for multiplication
__global__ void multiplyKernel(float *d_out, float value, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = value;
    
    // Each thread performs its own set of multiplications
    for(int i = 0; i < iterations; i++) {
        result *= value;
    }
    
    d_out[idx] = result;
}

// CPU version for comparison
void multiplyCPU(float *out, float value, int size, int iterations) {
    for(int idx = 0; idx < size; idx++) {
        float result = value;
        for(int i = 0; i < iterations; i++) {
            result *= value;
        }
        out[idx] = result;
    }
}

int main() {
    const int NUM_ELEMENTS = 1000000;
    const int ITERATIONS = 1000;
    const float VALUE = 1.001f;
    
    // Allocate host memory
    float *h_out_gpu = new float[NUM_ELEMENTS];
    float *h_out_cpu = new float[NUM_ELEMENTS];
    
    // Allocate device memory
    float *d_out;
    cudaMalloc(&d_out, NUM_ELEMENTS * sizeof(float));
    
    // Configure CUDA kernel
    int blockSize = 256;
    int numBlocks = (NUM_ELEMENTS + blockSize - 1) / blockSize;
    
    // GPU Implementation with timing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    multiplyKernel<<<numBlocks, blockSize>>>(d_out, VALUE, ITERATIONS);
    cudaDeviceSynchronize();
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;
    
    // Copy result back to host
    cudaMemcpy(h_out_gpu, d_out, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU Implementation with timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    multiplyCPU(h_out_cpu, VALUE, NUM_ELEMENTS, ITERATIONS);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    
    // Print results
    printf("GPU Time: %f seconds\n", gpu_time.count());
    printf("CPU Time: %f seconds\n", cpu_time.count());
    printf("Speedup: %fx\n", cpu_time.count() / gpu_time.count());
    
    // Verify results (check first and last elements)
    printf("First element GPU: %f, CPU: %f\n", h_out_gpu[0], h_out_cpu[0]);
    printf("Last element GPU: %f, CPU: %f\n", h_out_gpu[NUM_ELEMENTS-1], h_out_cpu[NUM_ELEMENTS-1]);
    
    // Cleanup
    delete[] h_out_gpu;
    delete[] h_out_cpu;
    cudaFree(d_out);
    
    return 0;
}
