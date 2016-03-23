#include "standard_flags.h"

DEFINE_string(save, "", "Where to save model to ?");
DEFINE_string(load, "", "Where to save model to ?");
DEFINE_int32(save_frequency_s, 60, "How often to save model (in seconds) ?");

DEFINE_string(solver,           "adadelta", "What solver to use (adadelta, sgd, adam, rmsprop, adagrad)");
DEFINE_double(learning_rate,    0.01,       "Learning rate for SGD and Adagrad.");
