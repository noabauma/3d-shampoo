"""3D-Shampoo with DeepSpeed pipeline (+ data) parallelism.

The model is wrapped in a DeepSpeed PipelineModule, whose topology tells the
optimizer which ranks form a data-parallel group. With N GPUs and
--stages S you get S pipeline stages and N/S data-parallel replicas; the
preconditioning work of each stage is distributed across its N/S replicas.

Run with:
    deepspeed ds_pp.py --deepspeed_config ds_config.json
    deepspeed ds_pp.py --deepspeed_config ds_config.json --stages 2   # needs >= 2 GPUs
"""

import argparse
import os
import sys
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import deepspeed
from deepspeed.pipe import PipelineModule

# loading the 3d-shampoo optimizer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import shampoo_3d

torch.manual_seed(42)


def main():
    parser = argparse.ArgumentParser(description='DeepSpeed pipeline parallelism with 3D-Shampoo.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--stages', type=int, default=1,
                        help='number of pipeline stages (must divide the number of GPUs)')
    parser.add_argument('--steps', type=int, default=20)
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    deepspeed.init_distributed()
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()

    hidden_dim = 8
    layers = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('fc1', nn.Linear(hidden_dim, hidden_dim)),
        ('relu1', nn.LeakyReLU()),
        ('fc2', nn.Linear(hidden_dim, hidden_dim)),
        ('relu2', nn.LeakyReLU()),
        ('fc3', nn.Linear(hidden_dim, hidden_dim)),
        ('relu3', nn.LeakyReLU()),
        ('fc4', nn.Linear(hidden_dim, hidden_dim)),
    ]))

    model = PipelineModule(layers=layers, loss_fn=F.mse_loss, num_stages=cmd_args.stages)

    optimizer = shampoo_3d.Shampoo_3D(params=model.parameters(),
                                      world_rank=world_rank,
                                      world_size=world_size,
                                      topology=model.topology(),
                                      lr=1e-1,
                                      momentum=0.9)

    if world_rank == 0:
        print("model topology:", model.topology())
        print("data-parallel groups:", model.topology().get_axis_comm_lists('data'))
        print("preconditioning ranks per parameter:", optimizer.partitioned_modules)

    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=model,
                                                         optimizer=optimizer)

    # a fixed toy regression batch; the PipelineEngine pulls micro-batches
    # from the iterator and moves them to the right device itself
    def batches():
        batchsize = model_engine.train_micro_batch_size_per_gpu()
        generator = torch.Generator().manual_seed(model_engine.grid.get_data_parallel_rank())
        x = torch.rand(batchsize, 2, hidden_dim // 2, generator=generator)
        t = torch.rand(batchsize, hidden_dim, generator=generator)
        while True:
            yield x, t

    data_iter = batches()
    for step in range(cmd_args.steps):
        loss = model_engine.train_batch(data_iter=data_iter)

        if world_rank == 0 and (step % 5 == 0 or step == cmd_args.steps - 1):
            print(f"step {step:3d}  loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
