# TP2: Deep Neural Networks from Scratch (PyTorch)

This folder contains the practical work and assignments for building, training, and analyzing multi-layer deep neural networks from scratch using PyTorch tensors (without high-level `torch.nn` modules).

## Contents

1. **`TP2_DeepNN_Step_by_Step_Class_activity.ipynb`**
   * **Overview:** A step-by-step implementation of an $L$-layer neural network using the row-major convention ($\mathbf{X}$ of shape $(m, n^{[0]})$).
   * **Key Components Implemented:**
     * Deep parameter initialization (`initialize_parameters_deep`) with layer-adapted scaling ($\sqrt{1 / n^{[l-1]}}$).
     * Forward propagation (`linear_forward`, `linear_activation_forward`, `L_model_forward`) supporting ReLU and Sigmoid activations.
     * Binary cross-entropy cost computation (`compute_cost`).
     * Backward propagation (`linear_backward`, `linear_activation_backward`, `L_model_backward`) and gradient descent updates (`update_parameters`).
     * Gradient verification against PyTorch's `autograd`.
   * **Exercises Completed:** Parameter counting across architectures, implementing Leaky ReLU forward/backward passes, and debugging backpropagation implementations.

2. **`TP2_Odd_or_Even_Assignment_DeepNetwork.ipynb`**
   * **Overview:** Applying the custom $L$-layer neural network toolbox to classify handwritten MNIST digits ($28 \times 28$ images) as **Odd ($1$)** or **Even ($0$)**.
   * **Key Tasks:**
     * **Prediction & Evaluation:** Implementing `predict` and `accuracy` metrics on a $10,000$-image training set, $2,000$-image dev set, and $10,000$-image test set.
     * **Training Loop:** Building `L_layer_model` to train architectures of arbitrary depth.
     * **Architecture Comparison:** Evaluating bias, variance, and learning dynamics across Logistic Regression (`[784, 1]`), a 2-layer network (`[784, 7, 1]`), and a 4-layer network (`[784, 20, 7, 5, 1]`).
     * **Initialization Analysis:** Observing vanishing gradients and stalled learning when using small uniform scaling (`scale=0.01`) in deep networks.
     * **Hyperparameter Tuning & Error Analysis:** Selecting the best model on the dev set ($\ge 97\%$ accuracy), evaluating on the test set, and analyzing per-digit error rates and misclassified images.

## Requirements

* Python 3.x
* `torch`
* `torchvision`
* `matplotlib`
