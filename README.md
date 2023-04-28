# 3D-Shampoo optimizer

3D-Shampoo is an distributed preconditioning based optimizer
to be used with the [DeepSpeed](https://github.com/microsoft/DeepSpeed) library.
Depending on the level of data parallelism of DeepSpeed, it automatically distributes the number of
preconditioning matrices across all available workers.

3D-Shampoo is a modified version of Google-Research's [Shampoo](https://github.com/noabauma/google-research/tree/master/scalable_shampoo/pytorch).
