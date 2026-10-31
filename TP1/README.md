# Notebook 1: Introduction to PyTorch

This foundational module introduces **PyTorch**, the primary deep learning library utilized throughout the semester. The focus is entirely on understanding and manipulating **tensors**—multi-dimensional arrays capable of hardware acceleration.

---

## Overview

The notebook guides you from basic tensor concepts to building optimized, vectorized operations ready for deep learning models. 

### Core Topics Covered
- **Part 0 · Setup:** Configuration of Google Colab environments with hardware acceleration (T4 GPU).
- **Part 1 · What is a tensor?:** Dimensionality (`ndim`), geometric hierarchy (scalars, vectors, matrices), and the standard **row convention** (one example per row).
- **Part 2 · Creating tensors:** Initializing tensors from Python data structures, NumPy arrays, and standard factory functions (`zeros`, `ones`, `arange`, `linspace`, `rand`, `randn`).
- **Part 3 · Data types:** Managing data types (`torch.float32`, `torch.int64`) and avoiding common performance bottlenecks like the *float64 trap*.
- **Part 4 · Operations:** Element-wise math vs. formal matrix multiplication (`@` / `torch.matmul`).
- **Part 5 · Aggregation:** Statistical metrics (`min`, `max`, `mean`, `sum`) and dimensional extraction (`argmax`, `argmin`).
- **Part 6 · Shape Manipulation:** Reshaping, memory views, concatenating/stacking, squeezing, and permuting spatial layouts (e.g., `HWC` $\leftrightarrow$ `CHW` format for images).
- **Part 7 · Indexing:** Slicing matrices and implementing boolean masking.
- **Part 8 · Broadcasting:** Rules governing multi-dimensional expansion and diagnosing invisible shape-mismatch logic bugs.
- **Part 9 · Memory & Interoperability:** Zero-copy memory sharing between PyTorch and NumPy (`torch.from_numpy`).
- **Part 10 · Reproducibility:** Managing deterministic pseudorandom pipelines with manual seeds.
- **Part 11 · GPU Acceleration:** Implementing device-agnostic code templates and benchmarking CPU vs. GPU operation speeds.

---

## Exercises Included

The notebook features **12 targeted exercises** evaluating core competencies:

1. **Matrix Multiplication:** Structural constraint checking and matrix transpositions.
2. **Deterministic Outputs:** Establishing exact replicability via CPU random state seeds.
3. **Hardware Semantics:** Exploring CUDA-specific seeding and runtime environments.
4. **Device Migration:** Initializing elements on the host CPU and staging them onto high-performance device contexts.
5. **Aggregation Bounds:** Locating boundary positions (`argmin`/`argmax`) within flattened arrays.
6. **Dimensional Squeezing:** Stripping singleton dimensions from higher-rank structures.
7. **Type Casting:** Mapping external NumPy arrays safely into exact 32-bit floating-point layers.
8. **Image Preprocessing:** Transform image batches from generic storage layout `(N, H, W, C)` to PyTorch-native tensor order `(N, C, H, W)` and flattening to standard ML matrices `(m, n)`.
9. **Broadcasting Analysis:** Predicting implicit tensor behavior purely from shape analysis.
10. **Feature Standardisation:** Vectorized normalization ($\mu=0, \sigma=1$) across dense columns without looping constructs.
11. **Debugging Shape Mismatches:** Correcting hidden broadcasting bugs within an explicit Mean Squared Error (MSE) loss function.
12. **Vectorization (Loop Replacement):** Rewriting sluggish nested Python loops into high-performance parallel operations for:
    - A standard linear layer layer computation ($Z = XW + b$)
    - An individual Min-Max column normalizer scaling metrics exactly to $[0, 1]$.

---

## Execution Instructions

### Local Environment Setup
To run this notebook locally, ensure you have Python 3 installed along with the required scientific computing libraries:

```bash
pip install torch numpy
```

### Google Colab Recommendations
For optimal performance and to complete Part 11, configure your Colab environment to utilize active hardware acceleration:
1. Navigate to **Runtime** $\rightarrow$ **Change runtime type**.
2. Select **T4 GPU** (or equivalent accelerator) from the hardware dropdown menu.
3. Click **Save** and execute the initialization script block.

---

## Next Steps
This lab prepares you for **Week 2: Logistic Regression as a Neural Network**, where the matrix multiplication, vectorization, and broadcasting skills learned here will be used to build a model from scratch.
