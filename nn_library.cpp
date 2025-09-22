#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <functional>

// A simple tensor class for multi-dimensional data and automatic differentiation.
class Tensor {
public:
    std::vector<double> data;
    std::vector<size_t> shape;
    std::vector<double> grad;
    Tensor* parent_a = nullptr;
    Tensor* parent_b = nullptr;
    std::function<void(const Tensor&)> backward_fn;
    bool requires_grad = false;

    // Constructor for a new tensor
    Tensor(const std::vector<double>& d, const std::vector<size_t>& s, bool req_grad = false)
        : data(d), shape(s), requires_grad(req_grad) {
        if (requires_grad) {
            grad.resize(data.size(), 0.0);
        }
    }

    // Helper to get the size of the tensor
    size_t size() const {
        size_t s = 1;
        for (size_t dim : shape) {
            s *= dim;
        }
        return s;
    }

    // Static function to create a zero-initialized tensor
    static Tensor zeros(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (size_t dim : shape) {
            size *= dim;
        }
        return Tensor(std::vector<double>(size, 0.0), shape);
    }

    // Static function to create a tensor with random values
    static Tensor rand(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (size_t dim : shape) {
            size *= dim;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1.0 / sqrt(shape.back()));
        std::vector<double> random_data(size);
        for (size_t i = 0; i < size; ++i) {
            random_data[i] = d(gen);
        }
        return Tensor(random_data, shape, true);
    }
};

// Abstract base class for all layers
class Layer {
public:
    virtual ~Layer() {}
    virtual Tensor forward(const Tensor& input) = 0;
    virtual void backward(const Tensor& grad_output) = 0;
    virtual void update_parameters(double lr) = 0;
    virtual std::vector<Tensor*> get_parameters() { return {}; }
};

// Linear layer implementation
class Linear : public Layer {
public:
    Tensor weights;
    Tensor bias;
    Tensor input_cache;

    Linear(size_t in_features, size_t out_features)
        : weights(Tensor::rand({out_features, in_features})),
          bias(Tensor::rand({out_features, 1})) {}

    Tensor forward(const Tensor& input) override {
        input_cache = input;
        size_t batch_size = input.shape[0];
        size_t in_features = input.shape[1];
        size_t out_features = weights.shape[0];

        // Ensure shapes are compatible for matrix multiplication
        if (input.shape[1] != weights.shape[1]) {
            throw std::runtime_error("Input and weights shape mismatch in Linear layer.");
        }

        std::vector<double> output_data(batch_size * out_features, 0.0);
        
        // Manual matrix multiplication
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                double val = 0.0;
                for (size_t k = 0; k < in_features; ++k) {
                    val += input.data[i * in_features + k] * weights.data[j * in_features + k];
                }
                output_data[i * out_features + j] = val + bias.data[j];
            }
        }
        Tensor output(output_data, {batch_size, out_features}, true);
        return output;
    }

    void backward(const Tensor& grad_output) override {
        // Compute gradients for this layer's parameters
        size_t batch_size = input_cache.shape[0];
        size_t in_features = input_cache.shape[1];
        size_t out_features = weights.shape[0];

        // Reset gradients
        std::fill(weights.grad.begin(), weights.grad.end(), 0.0);
        std::fill(bias.grad.begin(), bias.grad.end(), 0.0);

        // Gradient of weights: grad_output^T * input
        for (size_t j = 0; j < out_features; ++j) {
            for (size_t k = 0; k < in_features; ++k) {
                double grad_w = 0.0;
                for (size_t i = 0; i < batch_size; ++i) {
                    grad_w += grad_output.data[i * out_features + j] * input_cache.data[i * in_features + k];
                }
                weights.grad[j * in_features + k] = grad_w;
            }
        }

        // Gradient of bias: sum of grad_output across batch dimension
        for (size_t j = 0; j < out_features; ++j) {
            double grad_b = 0.0;
            for (size_t i = 0; i < batch_size; ++i) {
                grad_b += grad_output.data[i * out_features + j];
            }
            bias.grad[j] = grad_b;
        }

        // We don't need to propagate back to the input here, the model class handles it
    }

    void update_parameters(double lr) override {
        // Update weights and bias using gradient descent
        for (size_t i = 0; i < weights.data.size(); ++i) {
            weights.data[i] -= lr * weights.grad[i];
        }
        for (size_t i = 0; i < bias.data.size(); ++i) {
            bias.data[i] -= lr * bias.grad[i];
        }
    }

    std::vector<Tensor*> get_parameters() override {
        return {&weights, &bias};
    }
};

// ReLU activation function
class ReLU : public Layer {
public:
    Tensor input_cache;

    Tensor forward(const Tensor& input) override {
        input_cache = input;
        std::vector<double> output_data = input.data;
        for (double& val : output_data) {
            if (val < 0.0) {
                val = 0.0;
            }
        }
        Tensor output(output_data, input.shape, true);
        return output;
    }
    
    void backward(const Tensor& grad_output) override {
        // Compute gradients for this layer
        if (input_cache.requires_grad) {
            for (size_t i = 0; i < input_cache.data.size(); ++i) {
                if (input_cache.data[i] > 0.0) {
                    input_cache.grad[i] = grad_output.data[i];
                } else {
                    input_cache.grad[i] = 0.0;
                }
            }
        }
    }

    void update_parameters(double lr) override {
        // ReLU has no parameters to update
    }
};

// Simple Model class to combine layers
class Model {
public:
    std::vector<std::unique_ptr<Layer>> layers;

    void add_layer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Tensor forward(const Tensor& input) {
        Tensor current_tensor = input;
        for (const auto& layer : layers) {
            current_tensor = layer->forward(current_tensor);
        }
        return current_tensor;
    }

    void backward(const Tensor& grad_output) {
        Tensor current_grad = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            (*it)->backward(current_grad);
        }
    }

    void update_parameters(double lr) {
        for (const auto& layer : layers) {
            layer->update_parameters(lr);
        }
    }
};

// Cross-Entropy loss function for classification
Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target, Tensor& grad_output) {
    if (pred.size() != target.size() || pred.shape.back() != target.shape.back()) {
        throw std::runtime_error("Prediction and target shape mismatch in Cross-Entropy loss.");
    }
    
    size_t batch_size = pred.shape[0];
    size_t num_classes = pred.shape.back();

    // Compute softmax probabilities
    std::vector<double> softmax_data(pred.data.size());
    for (size_t i = 0; i < batch_size; ++i) {
        double max_val = pred.data[i * num_classes];
        for (size_t j = 1; j < num_classes; ++j) {
            if (pred.data[i * num_classes + j] > max_val) {
                max_val = pred.data[i * num_classes + j];
            }
        }
        double sum_exp = 0.0;
        for (size_t j = 0; j < num_classes; ++j) {
            softmax_data[i * num_classes + j] = std::exp(pred.data[i * num_classes + j] - max_val);
            sum_exp += softmax_data[i * num_classes + j];
        }
        for (size_t j = 0; j < num_classes; ++j) {
            softmax_data[i * num_classes + j] /= sum_exp;
        }
    }
    
    // Compute loss
    double loss_val = 0.0;
    for (size_t i = 0; i < pred.size(); ++i) {
        loss_val -= target.data[i] * std::log(softmax_data[i] + 1e-9); // Add epsilon for numerical stability
    }
    
    // Compute gradient for the backward pass
    std::vector<double> grad_data(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        grad_data[i] = softmax_data[i] - target.data[i];
    }
    grad_output = Tensor(grad_data, pred.shape);
    
    return Tensor({loss_val / batch_size}, {1}, true);
}


// Main function to demonstrate the library
int main() {
    try {
        // Build a simple neural network for MNIST classification
        const size_t INPUT_FEATURES = 784; // 28x28 image
        const size_t HIDDEN_FEATURES = 64;
        const size_t OUTPUT_FEATURES = 10; // Digits 0-9
        const size_t BATCH_SIZE = 1;

        Model model;
        model.add_layer(std::make_unique<Linear>(INPUT_FEATURES, HIDDEN_FEATURES));
        model.add_layer(std::make_unique<ReLU>());
        model.add_layer(std::make_unique<Linear>(HIDDEN_FEATURES, OUTPUT_FEATURES));

        // Create a hard-coded representation of a single hand-drawn digit '3'
        // This is a simplified representation of a 28x28 pixel image
        std::vector<double> x_data(INPUT_FEATURES, 0.0);
        // Drawing a crude '3'
        for (int i = 0; i < 28; ++i) {
            if (i > 5 && i < 22) { // Top line of the '3'
                x_data[5 * 28 + i] = 0.8;
                x_data[6 * 28 + i] = 0.8;
            }
            if (i > 5 && i < 22) { // Middle line
                x_data[13 * 28 + i] = 0.8;
                x_data[14 * 28 + i] = 0.8;
            }
            if (i > 5 && i < 22) { // Bottom line
                x_data[21 * 28 + i] = 0.8;
                x_data[22 * 28 + i] = 0.8;
            }
        }
        for (int i = 0; i < 28; ++i) { // Right vertical lines
            if (i > 5 && i < 15) {
                x_data[i * 28 + 21] = 0.8;
                x_data[i * 28 + 22] = 0.8;
            }
            if (i > 13 && i < 23) {
                x_data[i * 28 + 21] = 0.8;
                x_data[i * 28 + 22] = 0.8;
            }
        }
        
        // One-hot encoded target for the digit '3'
        std::vector<double> y_data(OUTPUT_FEATURES, 0.0);
        y_data[3] = 1.0;

        Tensor x_train(x_data, {BATCH_SIZE, INPUT_FEATURES}, true);
        Tensor y_train(y_data, {BATCH_SIZE, OUTPUT_FEATURES});

        // Training parameters
        double learning_rate = 0.01;
        int epochs = 100;

        std::cout << "Starting training on a single digit '3'..." << std::endl;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            Tensor y_pred = model.forward(x_train);

            // Calculate loss and get the gradient of the loss
            Tensor grad_output({0.0}, {0}); // Initialize with dummy values
            Tensor loss = cross_entropy_loss(y_pred, y_train, grad_output);

            // Backward pass
            model.backward(grad_output);

            // Update parameters
            model.update_parameters(learning_rate);

            if ((epoch + 1) % 10 == 0) {
                std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss.data[0] << std::endl;
            }
        }
        std::cout << "Training finished." << std::endl;

        // Make a prediction on the trained data
        Tensor prediction = model.forward(x_train);
        size_t predicted_class = 0;
        double max_prob = -1.0;
        for (size_t i = 0; i < prediction.size(); ++i) {
            if (prediction.data[i] > max_prob) {
                max_prob = prediction.data[i];
                predicted_class = i;
            }
        }
        std::cout << "\nAfter training, the predicted digit is: " << predicted_class << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
