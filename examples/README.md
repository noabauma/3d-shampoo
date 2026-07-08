# examples

Coding examples that show how 3D-Shampoo is used.

## simple_deepspeed_examples

The simplest possible setups, small MLPs trained on toy data. It helps to be familiar with
the DeepSpeed library.

* `ds_no_pp.py` — DeepSpeed data parallelism without a pipeline. The data-parallel topology
  is built by hand and given to the optimizer.
* `ds_pp.py` — the model is wrapped in a DeepSpeed `PipelineModule`, whose topology tells the
  optimizer which ranks form a data-parallel group. `--stages S` selects the number of
  pipeline stages (`S` must divide the number of GPUs).

Run them with the DeepSpeed launcher (see `launch.sh`):

```bash
deepspeed ds_no_pp.py --deepspeed_config ds_config.json
deepspeed ds_pp.py --deepspeed_config ds_config.json --stages 2   # needs >= 2 GPUs
```

With a single GPU the scripts still run, but everything is preconditioned by rank 0
(no distribution). Both scripts print the rank that preconditions each parameter and the
training loss, which should decrease.

## megatron_3D_parallelism

An example created by DeepSpeed itself and modified to work with the 3D-Shampoo optimizer to
fully utilize 3D parallelism (data + pipeline + tensor/model parallelism) on a GPT-2 style
model. To run this code you have to install Megatron-LM; section 6.6 of my MSc thesis
describes the steps to get 3D parallelism running and which libraries have to be installed.
`output_files/` contains the raw measurements from the thesis experiments
(file naming: `DP-MP-PP[rank,world_size].txt`).

Both examples were part of my MSc thesis.

**ATTENTION:** the Megatron example was written for the library versions used in the thesis
(2023) and is kept as a reference; newer DeepSpeed/Megatron-LM/PyTorch releases have changed
some of these APIs. The `simple_deepspeed_examples` were re-tested in 2026 with DeepSpeed 0.19
and torch 2.13.
