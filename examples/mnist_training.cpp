#include <iostream>
#include <vector>

#include "dali/core.h"
#include "dali/utils.h"

#include "dali/tensor/op/spatial.h"

using std::vector;
using std::string;

const std::string MNIST_PATH = "/home/sidor/tmp/";



/* Simple convnet to learn MNIST digit recognition
 *
 * MNIST_PATH must point a directory containing
 * two files:
 *    - mnistX.npy - mnist images in form of
 *                   (70000, 784) tensor of floats
 *                   with values between 0 and 1
 *    - mnistY.npy - mnist labels corresponding
 *                   this those images represented
 *                   as an int array of shape
 *                   (70000,)
 */

struct MnistCnn {
    ConvLayer conv1;
    ConvLayer conv2;
    Layer     fc;

    MnistCnn() {
        conv1 = ConvLayer(64, 1,   3, 3);
        conv2 = ConvLayer(128, 64, 3, 3);
        fc    = Layer(7 * 7 * 128, 10);
    }

    Tensor activate(Tensor images) const {
        // shape (B, 1, 28, 28)

        Tensor out = conv1.activate(images).relu();
        out = tensor_ops::max_pool(out, 2, 2);
        // shape (B, 64, 14, 14)

        out = conv2.activate(out).relu();
        out = tensor_ops::max_pool(out, 2, 2);
        // shape (B, 128, 7, 7)

        out = out.reshape({out.shape()[0], 7 * 7 * 128});
        out = fc.activate(out);
        // shape (B, 10)

        return out;
    }

    virtual std::vector<Tensor> parameters() const {
        return utils::concatenate({conv1.parameters(), conv2.parameters(), fc.parameters()});
    }
};

double accuracy(const MnistCnn& model, Tensor images, Tensor labels, int batch_size) {
    graph::NoBackprop nb;

    int num_images = images.shape()[0];

    auto num_correct = Array::zeros({}, DTYPE_INT32);
    for (int batch_start = 0; batch_start < num_images; batch_start += batch_size) {
        Slice batch_slice(batch_start, std::min(batch_start + batch_size, num_images));
        auto probs = model.activate(images[batch_slice]);
        Array predictions = op::astype(op::argmax(probs.w, -1), DTYPE_INT32);

        num_correct += op::sum(op::equals(predictions, (Array)labels.w[batch_slice]));
    }
    return (Array)(num_correct.astype(DTYPE_DOUBLE) / num_images);
}

std::vector<int> random_arange_int(int nelemens) {
    vector<int> res;
    res.reserve(nelemens);
    for (int i = 0; i < nelemens; ++i) res.push_back(i);
    std::random_shuffle(res.begin(), res.end());
    return res;
}


double training_epoch(const MnistCnn& model, std::shared_ptr<solver::AbstractSolver> solver,
                      Tensor images, Tensor labels, int batch_size) {
    int num_images = images.shape()[0];

    Tensor idxes = Tensor::empty({num_images}, DTYPE_INT32);
    idxes.w = random_arange_int(num_images);

    double epoch_error;

    auto params = model.parameters();

    for (int batch_start = 0; batch_start < images.shape()[0]; batch_start+=batch_size) {
        Tensor batch_idxes  = idxes[Slice(batch_start, std::min(batch_start + batch_size, num_images))];
        Tensor batch_images = images[batch_idxes];
        Tensor batch_labels = labels[batch_idxes];
        batch_images.constant = true;

        Tensor probs = model.activate(batch_images);

        Tensor error = tensor_ops::softmax_cross_entropy(probs, batch_labels);
        error.grad();
        epoch_error += (double)(Array)error.w.sum();

        graph::backward();
        solver->step(params);
    }

    return epoch_error / (double)num_images;
}


std::tuple<Tensor,Tensor,Tensor,Tensor> load_dataset() {

    auto data_x = Tensor::load(MNIST_PATH + "mnistX.npy");
    auto data_y_double = Tensor::load(MNIST_PATH + "mnistY.npy");
    auto data_y = Tensor(data_y_double.w.astype(DTYPE_INT32));
    data_x.constant = true;

    auto desired_data_x_shape = vector<int>{70000,784};
    ASSERT2(data_x.shape() == desired_data_x_shape, "wrong shape for mnist images.");
    ASSERT2(data_y.shape() == vector<int>{70000},   "wrong shape for mnist labels.");

    int num_images = data_x.shape()[0];
    Tensor idxes = Tensor::empty({num_images}, DTYPE_INT32);
    idxes.w = random_arange_int(num_images);
    data_x = data_x[idxes];
    data_y = data_y[idxes];
    data_x.constant = true;


    ELOG(data_x.shape());
    ELOG(data_y.shape());

    data_x = data_x.reshape({70000, 1, 28, 28});
    data_x.constant = true;

    Tensor train_x = data_x[Slice(0,     60000)];
    Tensor test_x  = data_x[Slice(60000, 70000)];

    Tensor train_y = data_y[Slice(0,     60000)];
    Tensor test_y  = data_y[Slice(60000, 70000)];

    train_x.constant = true;
    train_y.constant = true;

    test_x.constant = true;
    test_y.constant = true;
    return std::make_tuple(train_x, train_y, test_x, test_y);
}


int main() {
    utils::random::set_seed(123123);
    const int batch_size = 512;


    Tensor train_x, train_y, test_x, test_y;
    std::tie(train_x, train_y, test_x, test_y) = load_dataset();

    MnistCnn model;

    auto params = model.parameters();
    auto solver = solver::construct("adam", params, 0.0001);

    for (int i = 0; i < 20; ++i) {
        auto epoch_start_time = std::chrono::system_clock::now();
        auto epoch_error      = training_epoch(model, solver, train_x, train_y, batch_size);
        std::chrono::duration<double> epoch_duration
                = (std::chrono::system_clock::now() - epoch_start_time);
        auto test_acc  = accuracy(model, test_x, test_y, batch_size);

        std::cout << "Epoch " << i << ", train: " << epoch_error
                                   << ", test: "  << 100.0 * test_acc << '%'
                                   << ", time: " << epoch_duration.count() << "s" << std::endl;
    }
}
