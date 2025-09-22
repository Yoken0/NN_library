Neural Network from Scratch in C++

A hands-on project to build a foundational neural network library from the ground up using C++. This project demonstrates a deep understanding of the core concepts of deep learning, including tensors, layers, forward propagation, and backpropagation, without relying on external machine learning frameworks.
🚀 Key Features

    Custom Tensor Class: A fundamental building block for all operations, supporting multi-dimensional data and basic arithmetic.

    Modular Layers: Implemented core layers like Linear, ReLU, and Softmax as a flexible, extensible architecture.

    Manual Backpropagation: A custom backward pass implementation to calculate and propagate gradients through the network.

    Custom Loss Functions: Includes CrossEntropyLoss for classification tasks.

    Simple Training Loop: A complete example that trains a neural network to recognize a single hand-drawn digit.

    No External Dependencies: This library is built entirely with the C++ Standard Library, ensuring a lightweight and transparent implementation.

⚙️ How to Build and Run

This project uses a Makefile for a simple, automated build process.

    Clone the repository:

    git clone NN_library
    cd NN_library

    Build the executable:

    make

    Run the training program:

    ./nn_library

🧠 Project Components

The core of the library is structured around several C++ classes that mirror a standard deep learning framework:

    Tensor: Manages data and its shape, serving as the foundational data structure.

    Layer (Abstract Class): Defines a common interface for all network layers.

    Linear (Layer): Implements a fully-connected layer with learnable weights and biases.

    ReLU and Softmax (Layers): Provide essential non-linear activation functions.

    CrossEntropyLoss: Calculates the loss for the classification task.

    Model: The main class that orchestrates the layers and manages the forward and backward passes.

This project is an excellent demonstration of the foundational mechanics of deep learning and serves as a strong basis for more advanced implementations.
