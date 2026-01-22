<div align="center">
  <img src="logo.png" alt="sandbox-cuda" width="512"/>

  **🚀 CUDA benchmark comparing CPU vs GPU performance for parallel computations**

</div>

## Overview

A CUDA benchmark project that compares CPU vs GPU performance for repetitive multiplication operations. Demonstrates parallel computing capabilities with clear performance metrics.

## Features

- **CPU vs GPU comparison** - Side-by-side performance measurement
- **Configurable workload** - Adjust elements, iterations, and values
- **Result verification** - Validates GPU results against CPU reference
- **Clean benchmark output** - Clear timing and comparison data

## Quick Start

```bash
# Clone and build
git clone https://github.com/tsilva/sandbox-cuda.git
cd sandbox-cuda
mkdir build && cd build
cmake ..
make

# Run benchmark
./bin/multiply_benchmark
```

## Requirements

- CUDA Toolkit
- CMake 3.16+
- C++14 compatible compiler
- NVIDIA GPU (Maxwell or Volta/Turing architecture)

## Build Configuration

| Setting | Value |
|---------|-------|
| CMake minimum | 3.16 |
| C++/CUDA standard | C++14 |
| CUDA architectures | 52, 72 |
| Output directory | `bin/` |

## Architecture

```
src/multiply_benchmark.cu
├── multiplyKernel()  # CUDA kernel for parallel multiplication
├── multiplyCPU()     # Sequential CPU reference
└── main()            # Memory management, timing, verification
```

### Execution Configuration

- Block size: 256 threads
- Grid size: `(NUM_ELEMENTS + blockSize - 1) / blockSize`
- Memory: Host arrays + device allocation with cudaMemcpy

## License

MIT
