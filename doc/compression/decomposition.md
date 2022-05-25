# Decomposition

## Singular value decomposition (SVD)
In a CNN, the SVD can be applied to a fully connected layer, corresponding to a matrix $A$, resulting in the composition of two fully connected layers of lower dimension associated to the matrices $U'$ and $\Sigma' V^{T'}$. If $\Sigma$ is truncated to a rank $r=k$, dimensions change to $U'\in \mathbb{R}^{m \times k}$, $\Sigma' \in \mathbb{R}^{k \times k}$ and $V^{T'}\in \mathbb{R}^{k \times n}$.
Therefore, the SVD is applied to all weights of the linear layers, resulting in the decomposition of matrices. Subsequently, the weight matrices of the fully connected layers are compressed with a fixed rank $r$. Afterwards, the model is restructured, resulting in two smaller fully connected layers instead of one larger fully connected layer.

SVD-based compression (in combination with pruning and clustering) can be invoked by:

    hannah-train  compression=full

The default `rank_compression` of SVD is
: 4

This has been implemented for TC-ResNets, which consist of one linear layer as the last layer of the network. Thus, the resulting network comprises of a sequential layer at the end, including one fully connected layer with $\Sigma'*V^{T'}$ and one fully connected layer with $U'$. If those partial matrices are multiplied, they ideally yield a close approximation to the original matrix $A$. Finally, fine-tuning is performed, by continuing the training process with this restructured, compressed network while the same optimizer and parameters are used as for the first training section. With this approach, the SVD-based compression is easily combined with QAT and pruning. Both methods are also applied to the restructured, compressed model during retraining.
The layer's name can be adjusted to add other models.
