#include <cstring>
#include <gflags/gflags.h>
#include <memory>

#include <dali/core.h>

#include "utils.h"

int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
    "RNN Kindergarden - Lesson 1 - Learning to add.");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    utils::update_device(FLAGS_device);

    // How many examples to put in dataset
    const int NUM_EXAMPLES = 100;
    // How many number to add.
    const int EXAMPLE_SIZE = 3;
    // How many iterations of gradient descent to run.
    const int ITERATIONS = 150;
    // What is the learning rate.
    double LR = 0.01;

    // Generate random examples, all the rows sum to number between 0 and 1.
    auto X = Tensor::uniform(0.0, 1.0 / EXAMPLE_SIZE, {NUM_EXAMPLES, EXAMPLE_SIZE});
    // this is our dataset - please do not
    // consider during backpropagation.
    X = tensor_ops::consider_constant(X);

    // Compute sums of elements for each example. This is what we would
    // like the network to output.
    auto Y = X.sum(1);
    Y = tensor_ops::consider_constant(Y);

    // Those are our parameters: y_output = W1*X1 + ... + Wn*Xn
    // We initialize them to random numbers between 0 and 1.
    auto W = Tensor::uniform(0.0, 1.0, {EXAMPLE_SIZE, 1});

    W.print();

    std::vector<Tensor> params = { W };
    // auto solver = Solver::SGD<double>(params);

    for (int i = 0; i < ITERATIONS; ++i) {
        // What the network predicts the output will be.
        auto predY = X.dot(W).reshape({NUM_EXAMPLES});

        // Squared error between desired and actual output
        // E = sum((Ypred-Y)^2)
        auto error = ( (predY - Y) ^ 2 ).sum();
        // Mark error as what we compute error with respect to.
        error.grad();
        // Print error so that we know our progress.
        error.print();
        // Perform backpropagation algorithm.
        graph::backward();
        // Use gradient descent to update network parameters.
        // This is slightly obnoxious, but fear not - we
        // provide a solver class, so that you
        // never how to do it on your own!
        // solver.step(params, LR);
    }
    // Print the weights after we are done. The should all be close to one.
    W.print();
    W.dw.print();
}
