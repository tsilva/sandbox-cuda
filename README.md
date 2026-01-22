<div align="center">
  <img src="logo.png" alt="sandbox-cuda" width="256"/>

  # sandbox-cuda

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
  [![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C?logo=cmake)](https://cmake.org/)
  [![C++14](https://img.shields.io/badge/C%2B%2B-14-00599C?logo=cplusplus)](https://isocpp.org/)

  **Benchmark CPU vs GPU performance with parallel multiplication operations**

  [Overview](#overview) · [Quick Start](#quick-start) · [Configuration](#configuration)
</div>

---

## Overview

sandbox-cuda demonstrates the raw power difference between sequential CPU computation and massively parallel GPU execution. Run a million multiplication operations and see speedup factors that showcase why GPUs dominate parallel workloads.

### Features

- **Side-by-side comparison** - Identical workloads on CPU and GPU with timing
- **Configurable workload** - Adjust element count, iterations, and multiplier
- **Result verification** - Automatic validation that CPU and GPU produce matching results
- **Clean CMake build** - Simple build process with CUDA architecture targeting

## Quick Start

```bash
# Clone and build
git clone https://github.com/tsilva/sandbox-cuda.git
cd sandbox-cuda
mkdir build && cd build
cmake .. && make

# Run benchmark
../bin/multiply_benchmark
```

**Example output:**
```
GPU Time: 0.001234 seconds
CPU Time: 0.456789 seconds
Speedup: 370x
First element GPU: 2.718282, CPU: 2.718282
Last element GPU: 2.718282, CPU: 2.718282
```

## Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| NVIDIA GPU | CUDA-capable |
| CUDA Toolkit | 11.0+ |
| CMake | 3.16+ |
| C++ Compiler | C++14 support |

### Build from Source

```bash
mkdir build && cd build
cmake ..
make
```

The executable is placed in `bin/` at the project root.

## Configuration

Modify these constants in `src/multiply_benchmark.cu` to experiment with different workloads:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_ELEMENTS` | 1,000,000 | Number of parallel computations |
| `ITERATIONS` | 1,000 | Multiplications per element |
| `VALUE` | 1.001f | The floating-point multiplier |

### CUDA Execution

- **Block size**: 256 threads per block
- **Grid size**: Calculated as `(NUM_ELEMENTS + blockSize - 1) / blockSize`
- **Target architectures**: Maxwell (52) and Volta/Turing (72)

## Architecture

```
sandbox-cuda/
├── src/
│   └── multiply_benchmark.cu    # Main benchmark code
├── bin/                         # Build output
├── CMakeLists.txt               # Build configuration
└── README.md
```

The benchmark consists of three components:

1. **GPU Kernel** (`multiplyKernel`) - CUDA kernel performing parallel multiplications across thread blocks
2. **CPU Implementation** (`multiplyCPU`) - Sequential reference for comparison
3. **Main** - Orchestrates memory allocation, execution, timing, and verification

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.
