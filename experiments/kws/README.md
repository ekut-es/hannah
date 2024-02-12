# Example of using new FlexKI hannah search spaces for new models

The new search spaces for hannah are supposed to allow an easy expression of very large variability of neural network searches. 

This is a very simple example of using a it for searching a 1D-Convolutional Network 

There are 3 options to run the training in this folder. 

### Normal Neural Network Training

```bash
hannah-train
```

this will train a single neural network


### NAS on a flexible search space definition

```bash
hannah-train +experiment=ae_nas
```

This will use the flexible search space definition. Running for a direct nas on an aging evolution based optmizer. 

### Legacy NAS using fixed search spaces

```bash
hannah-train +experiment=legacy_nas
```

This uses the legacy/orginal Hannah search spaces as defined in the Paper. 