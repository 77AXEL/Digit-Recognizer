## 📋 Contents

This directory provides CPU optimized code:

### 1. **OpenBLAS Integration** (`libopenblas.dll`)
- Direct access to `dgemv` (matrix-vector multiplication)
- Multi-threaded BLAS operations with configurable thread count
- Significantly faster than NumPy for large matrix operations

### 2. **Optimized CNN Operations** (`optimized.cpp` → `optimized.dll`)
Highly optimized C++ implementations of:
- **ReLU**
- **Softmax**
- **Cross-Entropy Loss**
- **Gradient Descent**
- **L2 Regularization**

All functions are vectorized and optimized for native CPU architecture.

### 3. **Fast Vectors Addition** (`module.f90` → `module.X.pyd`)
Fortran-based vector addition:
- Faster than NumPy's `v1 + v2` for large arrays

## 📦 Dependencies

### Required
- Python 3.8+ with NumPy
- OpenBLAS library (included: `libopenblas.dll`)
- MinGW-w64 (for compilation)
- f2py (comes with NumPy)

### Optional
- `psutil` for automatic CPU core detection

```bash
pip install numpy psutil
```

## 🔨 Building from Source

### Automatic Build (Recommended)
```bash
# Windows
mingw32-make

# Linux
make
```

This compiles all components with maximum native optimizations:
- `-O3`: Highest optimization level
- `-march=native`: Use all CPU instructions available
- `-mtune=native`: Optimize for your specific CPU
- `-ffast-math`: Aggressive floating-point optimizations
- `-funroll-loops`: Loop unrolling for better performance

### Manual Build

**C++ Component:**
```bash
g++ -O3 -march=native -ffast-math -mtune=native -shared -o optimized.dll optimized.cpp
```

**Fortran Component:**
```bash
f2py -m module -c module.f90 --fcompiler=gnu95 --opt="-O3 -march=native -mtune=native -funroll-loops -ffast-math"
```