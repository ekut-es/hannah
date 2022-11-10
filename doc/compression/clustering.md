<!--
Copyright (c) 2022 University of Tübingen.

This file is part of hannah.
See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Clustering

## k-means
$k$-means clustering can be applied to the weights of all layers at the end of a training epoch. Each time, for a defined number of $k$ clusters per layer the centroids are identified per layer. Next, each weight is replaced with the value of the closest cluster center. The zero is not clustered to preserve the amount of zeros induced by pruning. This results in a total of $k+1$ distinct values. Additionally, at the end of the fitting process, clustering is applied. Afterwards, the weights are not further altered. After completing the training process, every value of each layer of the trained neural network should be equal to one cluster center or zero.

Clustering (in combination with pruning and SVD-based compression) can be invoked by:

    hannah-train  compression=full

The default `amount` of clusters is
: 15

If clustering or SVD was applied, the last checkpoint of the model during training was used for validation and testing the neural network, instead of the best checkpoint as previously used. Fixing the amount of epochs with which the final model is trained enables better comparability of the models for different compression parameters and ensures that the compression is sufficiently integrated into the training process. In particular the choice of a final model with high accuracy, which has not yet been compressed in the training process (via SVD or clustering) is avoided by ensuring that the last checkpoint is used. Consequently, fixing the amount of training epochs for the final model illuminates the effect that the compression during the training has on the model.

For applying clustering to quantized models, the batch norm fusion must be considered. This is not yet implemented.
