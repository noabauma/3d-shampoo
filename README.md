# 3D-Shampoo optimizer

3D-Shampoo is an distributed preconditioning based optimizer
to be used with the [DeepSpeed](https://github.com/microsoft/DeepSpeed) library.
Depending on the level of data parallelism of DeepSpeed, it automatically distributes the number of
preconditioning matrices across all available workers.

3D-Shampoo is a modified version of Google-Research's [Shampoo](https://github.com/noabauma/google-research/tree/master/scalable_shampoo/pytorch).

This code was created as part of my Master thesis "Distribtued Gradient Preconditioning for Training Large-Scale Models".

For more informations about 3D-Shampoo check out my Master thesis which is publicly available at the [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/615331).

The pseudocode of 3D-Shampoo is shown below

![image info](./figures/3d-shampoo_pseudocode.png)

3D-Shampoo will distributed the preconditioning matrices accordingly on the level of parallelism of DeepSpeed is active

![image info](./figures/different_levels_of_parallelism.png)

Not that ZeRO optimization is not supported due to storing the preconditioning matrices. Future update will maybe support this.
If there are more layers to precondition than number of available GPUs, the layers will be distributed accordingly to an own defined expected cost function.
If there are more GPUs than layers, #GPU - #layers will idle during preconditioning.

## How to install and use

Atm, you don't have to install it, you only need to link the folders to your python script.
You can use 3D-Shampoo like every other type of PyTorch based optimizers. 
3D-Shampoo will work if initialized with DeepSpeed, otherwise it is just basic Shampoo from Google-Research.

```python
# loading libraries
import torch
import torch.distributed as dist
import deepspeed
...

# loading 3d-shampoo optimizer
import sys
sys.path.append('../3d-shampoo/src/')
import shampoo

# initialize torch.distributed, define model, load datasets, etc.
...

optimizer = shampoo.Shampoo(params=model.parameters(),
                            world_rank=world_rank,
                            world_size=world_size,
                            topology=model.topology(), 
                            shapes=[tuple(p.shape) for p in model.parameters() if p.requires_grad], 
                            lr=1e-1, 
                            momentum=0.9, 
                            hyperparams=shampoo.ShampooHyperParams(ignore_embedding_layer=True))
							
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     optimizer=optimizer
                                                     )
														
# train your model
...
```

