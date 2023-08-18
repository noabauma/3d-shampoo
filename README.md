# 3D-Shampoo optimizer

3D-Shampoo is an distributed preconditioning based optimizer
to be used with the [DeepSpeed](https://github.com/microsoft/DeepSpeed) library.
Depending on the level of data parallelism of DeepSpeed, it automatically distributes the number of
preconditioning matrices across all available workers.

3D-Shampoo is a modified version of Google-Research's [Shampoo](https://github.com/noabauma/google-research/tree/master/scalable_shampoo/pytorch).

This code was created as part of my Master thesis "Distribtued Gradient Preconditioning for Training Large-Scale Models".

For more informations about 3D-Shampoo check out my Master thesis which is publicly available at the [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/615331).

The pseudocode of 3D-Shampoo is shown below

![image info](./3d-shampoo_pseudocode.png)

## How to install

TODO

## How to use

TODO
