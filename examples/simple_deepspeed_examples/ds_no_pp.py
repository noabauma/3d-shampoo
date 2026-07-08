"""3D-Shampoo with DeepSpeed data parallelism (no pipeline).

A plain model (no PipelineModule) has no DeepSpeed topology, so we build a
pure data-parallel topology by hand and give it to the optimizer. Every rank
then preconditions only its share of the layers and receives the remaining
updated parameters from the other ranks.

Run with:
    deepspeed ds_no_pp.py --deepspeed_config ds_config.json
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
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology

# loading the 3d-shampoo optimizer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import shampoo_3d

torch.manual_seed(42)


def main():
    parser = argparse.ArgumentParser(description='DeepSpeed data parallelism with 3D-Shampoo.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--steps', type=int, default=20)
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    deepspeed.init_distributed()
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()

    hidden_dim = 8
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('fc1', nn.Linear(4, hidden_dim)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_dim, hidden_dim)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_dim, hidden_dim)),
        ('relu3', nn.ReLU()),
        ('fc4', nn.Linear(hidden_dim, 2)),
    ]))

    # without a pipeline every rank is a pure data-parallel worker
    topology = PipeDataParallelTopology(num_pp=1, num_dp=world_size) if world_size > 1 else None

    optimizer = shampoo_3d.Shampoo_3D(params=model.parameters(),
                                      world_rank=world_rank,
                                      world_size=world_size,
                                      topology=topology,
                                      lr=1e-1,
                                      momentum=0.9)

    if world_rank == 0:
        print("preconditioning ranks per parameter:", optimizer.partitioned_modules)

    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=model,
                                                         optimizer=optimizer)

    # a fixed toy regression batch (different on every rank)
    batchsize = model_engine.train_micro_batch_size_per_gpu()
    generator = torch.Generator().manual_seed(world_rank)
    x = torch.rand(batchsize, 2, 2, generator=generator).to(model_engine.device)
    t = torch.rand(batchsize, 2, generator=generator).to(model_engine.device)

    for step in range(cmd_args.steps):
        y = model_engine(x)
        loss = F.mse_loss(y, t)

        model_engine.backward(loss)
        model_engine.step()

        if world_rank == 0 and (step % 5 == 0 or step == cmd_args.steps - 1):
            print(f"step {step:3d}  loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
