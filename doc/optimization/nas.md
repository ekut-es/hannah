# Neural architecture search

In contrast to hyperparameter optimization neural architecture search, explores new neural network hyperparameters.

A aging evolution based neural architecture search has been implemented as a hydra plugin:

   python -m speech_recognition.train --config-name config_unas

To launch multiple configuration jobs in parallel use joblib launcher:

    python -m speech_recognition.train --config-name config_unas hydra/launcher=joblib

Parametrization for neural architecture search need to be given as *YAML* configuration files at
the moment. For an example see: `speech_recognition/conf/config_unas.yaml`

## Parametrization

The Parametrization contains the following elements:

### Choice Parameter

Choice Parameters select options from a list of parameters. They are configured as a list of options in
the parameters. Example:

    conv_size: [1,3,5,7,9,11]

### Choice List Parameters

Choice List Parameters represent a variable length list of Choices. They are configured with the follwing parameters:

`min`
: Minimum length of list

`max`
: Maximum length of list+1

`choices`
: List of Choices

Example:

    min: 4
    max: 10
    choices:
      - _target_: "torch.nn.Conv2d"
        size: 3
      - _target : "torch.nn.MaxPool2d"
        size: 7

*Warning*: Mutations for intervall parameters currently always sample randomly from the range of values

### Intervall Parameters

Intervall Parameters represent a Scalar Value from an intervall of Values
They are configure with the following parameters:

`lower`
: lower bound of intervall [lower, upper[

`upper`
: upper bound of intervall [lower, upper[

`int`
: set to true to generate integers

`log`
: set to true to generate log scaled distribution

### Subset Parameter

Subset Parameters select a subset of a list of choices.

They are configured using the following parameters:

`choices`
: List of choices to sample from


`size`
: size of the subset to generate



### Partition Parameter

Partition parameters split the list of choices into a predefined number of partitions.

They are configured using the following parameters:

`choices`
: List of choices to partition

`partition`
: Number of partitions to generate
