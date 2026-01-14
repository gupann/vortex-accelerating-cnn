# CNN Inference and Tensor Core Acceleration on Vortex

This project implements an end-to-end convolutional neural network (CNN) inference pipeline on the **Vortex RISC-V GPGPU**, evaluates its correctness and performance, and explores **tensor core (TCU) acceleration** using Vortex’s WMMA API at the kernel level.

The work is divided into two parts:

1. A **fully functional CNN inference pipeline** running on Vortex GPU kernels.  
2. A **tensor-core-accelerated SGEMM kernel** used to evaluate the performance potential of WMMA for CNN workloads.

**Slides:** [Presentation](https://docs.google.com/presentation/d/16v4S7cEtVrpoikDRdTHzFyBff66ID1WVYZXL_87Vwjo/edit?usp=sharing)

---

## Project Overview

### CNN Architecture

The implemented CNN matches the following architecture:

Input (1×28×28)
→ Conv2D (8 filters, 3×3, stride=1, valid)
→ MaxPool2D (2×2)
→ ReLU
→ Flatten
→ Fully Connected (10)
→ Softmax

- Dataset: **Fashion-MNIST**
- Training framework: **Keras / TensorFlow**
- Inference runtime: **Vortex GPU + CPU**
- Precision: **FP32**


---

## CNN Inference Pipeline (`demo/CNN`)

### Execution

The CNN pipeline runs the convolution, pooling, and ReLU layers on the **Vortex GPU**, while the final fully connected layer runs on the **CPU**.

To build and run:

```bash
make -C demo/CNN
make -C demo/CNN run-simx
```

To sweep GPU configurations (baseline performance):

```bash
./ci/blackbox.sh --driver=simx --app=CNN --cores=4 --warps=4 --threads=4 --args="-n64"
./ci/blackbox.sh --driver=simx --app=CNN --cores=4 --warps=4 --threads=8 --args="-n64"
./ci/blackbox.sh --driver=simx --app=CNN --cores=4 --warps=8 --threads=4 --args="-n64"
./ci/blackbox.sh --driver=simx --app=CNN --cores=4 --warps=8 --threads=8 --args="-n64"
```

The application reports:
- Per-image correctness
- Overall accuracy
- GPU performance metrics (instructions, cycles, IPC)

Correctness Validation
--------------------------

Correctness is validated by:

*   Matching CNN architecture exactly between Keras and Vortex
*   Exporting trained weights and test images
*   Comparing layer-level statistics:
    *   min / max / mean
    *   sample values (y=0, x=0, channels 0–7)
*   Running inference on 100 test images
  
The Vortex CNN pipeline achieves accuracy consistent with the Keras reference model.

Tensor Core (WMMA) Acceleration Exploration
-----------------------------------------------

### What Was Implemented

Tensor core acceleration was explored using **Vortex’s WMMA (warp-level matrix multiply–accumulate) API** in the regression framework:

*   Location: tests/regression/sgemm\_tcu
*   Uses:
    *   vortex::tensor::wmma\_context
    *   load\_matrix\_sync
    *   mma\_sync
    *   store\_matrix\_sync
        
*   Precision:
    *   FP16 inputs
    *   FP32 accumulation
        
*   Performance metrics:
    *   Instructions      
    *   Cycles
    *   IPC

### Purpose

This kernel serves as a **proxy for im2col-based convolution**, allowing evaluation of the performance benefits of tensor cores without fully rewriting the CNN convolution kernel.

### Important Clarification

⚠️ The WMMA tensor-core kernel is **not integrated into the end-to-end CNN pipeline**.The CNN inference uses baseline GPU convolution kernels, while WMMA is evaluated **independently** to estimate acceleration potential.

This separation ensures correctness and simplifies performance analysis.

Key Results
---------------

*   The CNN pipeline runs correctly on Vortex GPU
*   Layer-by-layer outputs match Python reference
*   Accuracy over 100 test images matches Keras expectations
*   Tensor-core SGEMM shows **significantly higher IPC** compared to scalar/FPU convolution kernels
*   Performance sweeps demonstrate sensitivity to warp and thread configuration
    

Reproducing Results
-----------------------

### Requirements

*   Vortex toolchain
*   Verilator
*   RISC-V GNU toolchain
*   Python 3 + TensorFlow/Keras (for training/export)
    

### Steps

1.  Train and export data
   ```bash
cd training
python cnn.py
python export_weights.py
python export_images.py

```
    
2.  Build Vortex:
  ```bash
mkdir build && cd build
../configure
make

```
    
5.  Run CNN inference:
   ```bash
make -C demo/CNN run-simx
```
