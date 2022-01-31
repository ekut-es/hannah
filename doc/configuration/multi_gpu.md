# Multi GPU training

Hannah supports multi GPU-Training using the lightning distributed APIs:

We provide preset trainer configs for distributed data parallel training:

```hannah-train trainer=ddp trainer.gpus=[0,1]```


And for sharded training using fairscale:


```hannah-train trainer=sharded trainer.gpus=[0,1]```


Sharded training distributes some of the model parameters across multiple GPUs and allows fitting bigger models in the same amount of GPU memory.
