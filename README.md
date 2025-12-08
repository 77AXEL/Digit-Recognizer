# Digit Recognition CNN from Scratch

A handwritten digit recognition system built from scratch using Python, with performance-optimized implementations using C extensions, Fortran, and custom DLLs.
<p align="center">
  <img src="assets/animation.gif">
</p>

## 📋 Overview

This is a **learning and experimentation project** that implements a Convolutional Neural Network (CNN) for digit recognition without relying on high-level deep learning frameworks. The project explores low-level optimization techniques and demonstrates how to integrate compiled code with Python for better performance.

## 🎯 Key Features

- **From-scratch implementation** using NumPy and SciPy
- **Performance optimizations** through multiple approaches:
  - C extensions for core functions (softmax, ReLU, loss calculation, weight/bias updates)
  - Fortran f2py extension for vector addition operations
  - OpenBLAS integration for accelerated tensor dot products
- **Custom compiled modules** loaded via ctypes
- **Complete training pipeline** for digit classification

## 🗂️ Project Structure

```
├── assets/
│   └── animation.gif          # Neural network visualization
├── lib/
│   ├── funcs.cpp              # C++ source for custom functions
│   ├── funcs.dll              # Compiled C DLL (softmax, ReLU, loss, updates)
│   ├── funcs.exp
│   ├── funcs.lib
│   ├── funcs.pdb
│   ├── libgcc_s_seh-1.dll     # GCC runtime library
│   ├── libgfortran-5.dll      # Fortran runtime library
│   ├── libopenblas.dll        # OpenBLAS for tensor operations
│   ├── libquadmath-0.dll      # Quadmath library
│   ├── libmodule.V4URNAUZAJSVN6KEXZGZ...dll  # Compiled Fortran DLL
│   ├── module.cp38-win_amd64.pyd  # Fortran f2py compiled module
│   └── module.f90                 # Fortran source for vector operations
├── models/
│   └── model.bin              # Saved model weights
├── data.zip                   # Training dataset (needs extraction)
├── gitattributes.txt
├── lab.ipynb                  # Main training notebook
├── LICENSE
├── modules.py                 # Python module definitions
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- SciPy
- Jupyter Notebook (for running lab.ipynb)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/77AXEL/Digit-Recognizer
cd Digit-Recognizer
```

2. **Extract the dataset**:
```bash
unzip data.zip
```
This step is required before training the model.

3. Install Python dependencies:
```bash
pip install numpy scipy jupyter
```

### Running the Project

Open and run the main notebook:
```bash
jupyter notebook lab.ipynb
```

The notebook contains the complete pipeline for training and testing the digit recognition CNN.

## 🔧 Technical Details

### Performance Optimizations

1. **OpenBLAS Integration**: Replaces NumPy's default BLAS for faster tensor dot products
2. **C Extensions**: Critical functions implemented in C and loaded via ctypes:
   - Softmax activation
   - ReLU activation
   - Loss computation
   - Weight and bias updates
3. **Fortran Extensions**: Vector addition optimized with f2py (`module.sum_vectors`)

### Architecture

The project uses a convolutional neural network architecture implemented from scratch with custom forward and backward propagation logic.

## 📊 Dataset

The training dataset is provided in `data.zip` and must be extracted before use. The dataset contains handwritten digit images for training and validation.

## 🎓 Learning Objectives

This project demonstrates:
- Low-level neural network implementation
- Integration of multiple programming languages (Python, C++, Fortran)
- Performance optimization techniques
- Understanding of CNN fundamentals without frameworks

## 📝 License

See the `LICENSE` file for details.

## ⚠️ Note

This is an educational project focused on understanding the internals of neural networks and optimization techniques. For production use, consider established frameworks like TensorFlow or PyTorch.

---

**Built with curiosity and compiled with care** 🧠✨
