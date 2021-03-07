# Torchmobile backend

A backend that runs the trained neural network through torchmobile on the local cpu.

This is an example implementation of backend that allows testing of backend integrations without installation of further packages.
It should not be used for serious optimizations, as measurements are much to noisy.


## Configuration

val_batches
:1 (number of batches used for validation)

test_batches
:1 (number of batches used for test)

val_frequency
:10 (run backend every n validation epochs)
