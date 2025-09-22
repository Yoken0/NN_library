#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <random>
#include <chrono>

// --- Transpose function for Tensors ---
class Tensor; // Forward declaration
Tensor transpose(const Tensor& input);

// === Tensor Class ===
// A simple multi-dimensional array with automatic differentiation capabilities.
class Tensor {
public:
    std::vector<double> data;
    std::vector<int> shape;
    std::vector<double> grad;
    std::shared_ptr<Tensor> operand_a;
    std::shared_ptr<Tensor> operand_b;
    std::function<void()> backward_op;

    // Default constructor to allow for creation without immediate data
    Tensor() : data({}), shape({}), grad({}) {}

    Tensor(std::vector<double> d, std::vector<int> s) : data(d), shape(s) {
        if (!check_shape()) {
            throw std::runtime_error("Data size does not match shape.");
        }
        grad.resize(data.size(), 0.0);
    }

    bool check_shape() const {
        size_t total_elements = 1;
        for (int dim : shape) {
            total_elements *= dim;
        }
        return total_elements == data.size();
    }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0);
        if (operand_a) operand_a->zero_grad();
        if (operand_b) operand_b->zero_grad();
    }

    void backward(double upstream_grad = 1.0) {
        if (data.empty()) return;
        
        for (size_t i = 0; i < grad.size(); ++i) {
            grad[i] += upstream_grad;
        }

        if (backward_op) {
            backward_op();
        }
    }
};

// --- Operators for Tensors ---
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size()) {
        throw std::runtime_error("Tensor sizes must match for addition.");
    }
    std::vector<double> result_data(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) {
        result_data[i] = a.data[i] + b.data[i];
    }
    Tensor result(result_data, a.shape);
    result.operand_a = std::make_shared<Tensor>(a);
    result.operand_b = std::make_shared<Tensor>(b);
    result.backward_op = [&, result_grad = result.grad]() {
        for (size_t i = 0; i < a.grad.size(); ++i) {
            result.operand_a->grad[i] += result_grad[i];
            result.operand_b->grad[i] += result_grad[i];
        }
    };
    return result;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size()) {
        throw std::runtime_error("Tensor sizes must match for subtraction.");
    }
    std::vector<double> result_data(a.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) {
        result_data[i] = a.data[i] - b.data[i];
    }
    Tensor result(result_data, a.shape);
    result.operand_a = std::make_shared<Tensor>(a);
    result.operand_b = std::make_shared<Tensor>(b);
    result.backward_op = [&, result_grad = result.grad]() {
        for (size_t i = 0; i < a.grad.size(); ++i) {
            result.operand_a->grad[i] += result_grad[i];
            result.operand_b->grad[i] -= result_grad[i];
        }
    };
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2 || a.shape[1] != b.shape[0]) {
        throw std::runtime_error("Invalid shapes for matrix multiplication.");
    }

    int rows_a = a.shape[0];
    int cols_a = a.shape[1];
    int cols_b = b.shape[1];

    std::vector<double> result_data(rows_a * cols_b, 0.0);
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            for (int k = 0; k < cols_a; ++k) {
                result_data[i * cols_b + j] += a.data[i * cols_a + k] * b.data[k * cols_b + j];
            }
        }
    }

    Tensor result(result_data, {rows_a, cols_b});
    result.operand_a = std::make_shared<Tensor>(a);
    result.operand_b = std::make_shared<Tensor>(b);
    result.backward_op = [&, result_grad = result.grad]() {
        // dL/dA = dL/dZ * B^T
        std::vector<double> grad_a_data(a.data.size(), 0.0);
        for (int i = 0; i < rows_a; ++i) {
            for (int j = 0; j < cols_a; ++j) {
                for (int k = 0; k < cols_b; ++k) {
                    grad_a_data[i * cols_a + j] += result_grad[i * cols_b + k] * b.data[j * cols_b + k];
                }
            }
        }
        // dL/dB = A^T * dL/dZ
        std::vector<double> grad_b_data(b.data.size(), 0.0);
        for (int i = 0; i < cols_a; ++i) {
            for (int j = 0; j < cols_b; ++j) {
                for (int k = 0; k < rows_a; ++k) {
                    grad_b_data[i * cols_b + j] += a.data[k * cols_a + i] * result_grad[k * cols_b + j];
                }
            }
        }
        for (size_t i = 0; i < grad_a_data.size(); ++i) result.operand_a->grad[i] += grad_a_data[i];
        for (size_t i = 0; i < grad_b_data.size(); ++i) result.operand_b->grad[i] += grad_b_data[i];
    };
    return result;
}

Tensor sigmoid(const Tensor& x) {
    std::vector<double> result_data(x.data.size());
    for (size_t i = 0; i < x.data.size(); ++i) {
        result_data[i] = 1.0 / (1.0 + std::exp(-x.data[i]));
    }
    Tensor result(result_data, x.shape);
    result.operand_a = std::make_shared<Tensor>(x);
    result.backward_op = [&, result_grad = result.grad]() {
        for (size_t i = 0; i < result_grad.size(); ++i) {
            double s = result.data[i];
            result.operand_a->grad[i] += result_grad[i] * s * (1.0 - s);
        }
    };
    return result;
}

Tensor relu(const Tensor& x) {
    std::vector<double> result_data(x.data.size());
    for (size_t i = 0; i < x.data.size(); ++i) {
        result_data[i] = std::max(0.0, x.data[i]);
    }
    Tensor result(result_data, x.shape);
    result.operand_a = std::make_shared<Tensor>(x);
    result.backward_op = [&, result_grad = result.grad]() {
        for (size_t i = 0; i < result_grad.size(); ++i) {
            result.operand_a->grad[i] += result_grad[i] * (result.data[i] > 0 ? 1.0 : 0.0);
        }
    };
    return result;
}

// === Layers ===
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output_grad) = 0;
    virtual void update_params(double learning_rate) = 0;
};

class Linear : public Layer {
public:
    Tensor weights;
    Tensor bias;
    Tensor input_cache;

    Linear(int input_dim, int output_dim) {
        // Initialize weights and bias with random values
        std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / (input_dim + output_dim)));
        std::vector<double> w_data(input_dim * output_dim);
        for (size_t i = 0; i < w_data.size(); ++i) {
            w_data[i] = dist(rng);
        }
        weights = Tensor(w_data, {input_dim, output_dim});
        bias = Tensor(std::vector<double>(output_dim, 0.0), {1, output_dim});
    }

    Tensor forward(const Tensor& input) override {
        input_cache = input;
        Tensor matmul_result = matmul(input, weights);
        return add(matmul_result, bias);
    }

    Tensor backward(const Tensor& output_grad) override {
        // Compute gradients for weights and bias
        Tensor weights_t = transpose(weights);
        
        // This is a simplified backward calculation without a computational graph.
        Tensor grad_weights_from_output_grad = matmul(transpose(input_cache), output_grad);
        weights.grad = grad_weights_from_output_grad.data;
        bias.grad = output_grad.data;

        // Compute gradient for the input and return it
        Tensor grad_input = matmul(output_grad, weights_t);
        return grad_input;
    }

    void update_params(double learning_rate) override {
        for (size_t i = 0; i < weights.data.size(); ++i) {
            weights.data[i] -= learning_rate * weights.grad[i];
        }
        for (size_t i = 0; i < bias.data.size(); ++i) {
            bias.data[i] -= learning_rate * bias.grad[i];
        }
    }
};

class ReLU : public Layer {
public:
    Tensor input_cache;

    ReLU() = default;

    Tensor forward(const Tensor& input) override {
        input_cache = input;
        return relu(input);
    }

    Tensor backward(const Tensor& output_grad) override {
        // The gradient of ReLU is 1 for positive inputs, 0 otherwise.
        std::vector<double> grad_input_data(input_cache.data.size());
        for (size_t i = 0; i < input_cache.data.size(); ++i) {
            grad_input_data[i] = input_cache.data[i] > 0 ? output_grad.data[i] : 0.0;
        }
        return Tensor(grad_input_data, input_cache.shape);
    }

    void update_params(double learning_rate) override {
        // No parameters to update for ReLU
    }
};

class Softmax : public Layer {
public:
    Tensor output_cache;
    
    Softmax() = default;

    Tensor forward(const Tensor& input) override {
        // Softmax implementation
        std::vector<double> exp_data(input.data.size());
        double sum_exp = 0.0;
        for (size_t i = 0; i < input.data.size(); ++i) {
            exp_data[i] = std::exp(input.data[i]);
            sum_exp += exp_data[i];
        }

        std::vector<double> result_data(input.data.size());
        for (size_t i = 0; i < input.data.size(); ++i) {
            result_data[i] = exp_data[i] / sum_exp;
        }

        output_cache = Tensor(result_data, input.shape);
        return output_cache;
    }

    Tensor backward(const Tensor& output_grad) override {
        // Since CrossEntropyLoss's backward method already handles the combined
        // gradient of softmax and the loss, this layer's backward can be simplified.
        return Tensor(std::vector<double>(output_grad.data.size(), 0.0), output_grad.shape);
    }

    void update_params(double learning_rate) override {
        // No parameters to update for Softmax
    }
};

// --- Transpose function for Tensors ---
Tensor transpose(const Tensor& input) {
    if (input.shape.size() != 2) {
        throw std::runtime_error("Transpose is only supported for 2D tensors.");
    }
    int rows = input.shape[0];
    int cols = input.shape[1];
    std::vector<double> transposed_data(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed_data[j * rows + i] = input.data[i * cols + j];
        }
    }
    return Tensor(transposed_data, {cols, rows});
}

// === Loss Function ===
class MSELoss {
public:
    Tensor forward(const Tensor& predicted, const Tensor& target) {
        if (predicted.data.size() != target.data.size()) {
            throw std::runtime_error("Predicted and target tensors must have the same size.");
        }
        std::vector<double> diff(predicted.data.size());
        for (size_t i = 0; i < predicted.data.size(); ++i) {
            diff[i] = predicted.data[i] - target.data[i];
        }
        double sum_sq_diff = 0.0;
        for (double d : diff) {
            sum_sq_diff += d * d;
        }
        double loss_value = sum_sq_diff / predicted.data.size();
        return Tensor({loss_value}, {1});
    }

    Tensor backward(const Tensor& predicted, const Tensor& target) {
        std::vector<double> grad_data(predicted.data.size());
        for (size_t i = 0; i < predicted.data.size(); ++i) {
            grad_data[i] = 2.0 * (predicted.data[i] - target.data[i]) / predicted.data.size();
        }
        return Tensor(grad_data, predicted.shape);
    }
};

class CrossEntropyLoss {
public:
    Tensor forward(const Tensor& predicted_logits, const Tensor& target_one_hot) {
        if (predicted_logits.data.size() != target_one_hot.data.size()) {
            throw std::runtime_error("Predicted and target tensors must have the same size.");
        }
        double loss = 0.0;
        double sum_exp = 0.0;
        for(double val : predicted_logits.data) {
            sum_exp += std::exp(val);
        }
        
        for (size_t i = 0; i < predicted_logits.data.size(); ++i) {
            if (target_one_hot.data[i] > 0.5) { // Assuming one-hot encoded
                loss = -predicted_logits.data[i] + std::log(sum_exp);
                break;
            }
        }
        return Tensor({loss}, {1});
    }

    Tensor backward(const Tensor& predicted_logits, const Tensor& target_one_hot) {
        std::vector<double> grad_data(predicted_logits.data.size());
        double sum_exp = 0.0;
        for(double val : predicted_logits.data) {
            sum_exp += std::exp(val);
        }

        for (size_t i = 0; i < predicted_logits.data.size(); ++i) {
            double softmax_prob = std::exp(predicted_logits.data[i]) / sum_exp;
            grad_data[i] = softmax_prob - target_one_hot.data[i];
        }
        return Tensor(grad_data, predicted_logits.shape);
    }
};

// === Model ===
class Model {
public:
    std::vector<std::unique_ptr<Layer>> layers;

    void add_layer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Tensor forward(Tensor input) {
        Tensor output = input;
        for (const auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    void backward(const Tensor& output_grad) {
        Tensor current_grad = output_grad;
        for (size_t i = layers.size(); i > 0; --i) {
            current_grad = layers[i - 1]->backward(current_grad);
        }
    }

    void update_params(double learning_rate) {
        for (const auto& layer : layers) {
            layer->update_params(learning_rate);
        }
    }
};

// === Main Function ===
int main() {
    std::cout << "Starting Neural Network Training for a single digit..." << std::endl;

    // Hyperparameters
    const int INPUT_DIM = 784;  // 28x28 pixels
    const int HIDDEN_DIM = 128;
    const int OUTPUT_DIM = 10;  // Digits 0-9
    const double LEARNING_RATE = 0.001;
    const int EPOCHS = 1000;

    // Create a simple model
    Model model;
    model.add_layer(std::make_unique<Linear>(INPUT_DIM, HIDDEN_DIM));
    model.add_layer(std::make_unique<ReLU>());
    model.add_layer(std::make_unique<Linear>(HIDDEN_DIM, OUTPUT_DIM));
    
    CrossEntropyLoss loss_fn;
    Softmax softmax_layer;

    // Hard-coded input for the digit '3' (simplified)
    std::vector<double> input_data(INPUT_DIM, 0.0);
    // A very simple representation of a '3'
    for (int i = 0; i < 28; ++i) {
        if (i >= 5 && i <= 22) {
            input_data[i * 28 + 10] = 1.0;
            input_data[i * 28 + 11] = 1.0;
            input_data[i * 28 + 12] = 1.0;
            if (i > 8 && i < 14) {
                input_data[i * 28 + 13] = 1.0;
            }
        }
    }
    input_data[5 * 28 + 13] = 1.0;
    input_data[12 * 28 + 13] = 1.0;
    input_data[13 * 28 + 13] = 1.0;
    input_data[14 * 28 + 13] = 1.0;
    input_data[15 * 28 + 13] = 1.0;
    input_data[16 * 28 + 13] = 1.0;
    
    Tensor input_tensor(input_data, {1, INPUT_DIM});

    // Target for the digit '3' (one-hot encoded)
    std::vector<double> target_data(OUTPUT_DIM, 0.0);
    target_data[3] = 1.0;
    Tensor target_tensor(target_data, {1, OUTPUT_DIM});

    // Training loop
    for (int epoch = 0; epoch <= EPOCHS; ++epoch) {
        Tensor predicted_logits = model.forward(input_tensor);
        Tensor predicted_probs = softmax_layer.forward(predicted_logits);

        Tensor loss = loss_fn.forward(predicted_logits, target_tensor);

        if (epoch % 10 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss.data[0] << std::endl;
        }

        Tensor loss_grad = loss_fn.backward(predicted_logits, target_tensor);
        // This is a simplified backward pass without a computational graph.
        // The gradient from the loss function is the same as the gradient for the softmax input.
        model.backward(loss_grad);
        model.update_params(LEARNING_RATE);
    }

    // Final prediction
    Tensor final_output_logits = model.forward(input_tensor);
    Tensor final_output_probs = softmax_layer.forward(final_output_logits);
    int predicted_digit = 0;
    double max_prob = -1.0;
    for (int i = 0; i < final_output_probs.data.size(); ++i) {
        if (final_output_probs.data[i] > max_prob) {
            max_prob = final_output_probs.data[i];
            predicted_digit = i;
        }
    }

    std::cout << "\nTraining finished." << std::endl;
    std::cout << "Predicted digit: " << predicted_digit << std::endl;

    return 0;
}
