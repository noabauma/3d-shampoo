import os
import argparse
from collections import OrderedDict

import torch
import torch.distributed as dist


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import deepspeed

from deepspeed.profiling.flops_profiler import get_model_profile

torch.manual_seed(42)


def init_dist_process_group(backend='nccl'):
    if os.environ.get('LOCAL_RANK', None) is not None:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_size = int(os.environ.get('LOCAL_SIZE', world_size))
    elif os.environ.get('SLURM_JOBID', None) is not None:
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1
        local_size = 1

    if world_size > 1:
        assert dist.is_available()

        deepspeed.init_distributed(dist_backend=backend)

        assert dist.get_rank() == world_rank
        assert dist.get_world_size() == world_size

    return local_rank, local_size, world_rank, world_size


def main():
    local_rank, local_size, world_rank, world_size = init_dist_process_group()

    # only needed for Barry (multi GPU on one node)
    if local_size == world_size:
        torch.cuda.set_device(local_rank)

    node = os.environ.get('SLURMD_NODENAME', local_rank)

    print(f"Hello I am node: {node}, with world_rank {world_rank}, local_rank {local_rank} and world_size {world_size}")

    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=world_rank,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    print("cmd_args: ", cmd_args)

    batchsize = 2

    hidden_dim = 3
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
        
    
    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=model,
                                                        )


    #param_shapes = [p.shape for p in model_engine.parameters()]
    #print(param_shapes, "\n")

    for step in range(1):  #two epochs for nsys
        x = torch.rand(batchsize,2,2).cuda() + world_rank
        t = torch.rand(batchsize, 2).cuda()
        batch = (x, t)


        print("grads before:\n", [p.grad for p in model_engine.parameters()], "\n")
        
        #forward() method
        y = model_engine(x)

        loss = F.mse_loss(y, t)

        #runs backpropagation
        model_engine.backward(loss)

        print("grads after:\n", [p.grad for p in model_engine.parameters()], "\n")

        #weight update
        model_engine.step()

    """
    flops, macs, params = get_model_profile(model=model_engine, # model
                                    input_shape=(batchsize, 2,2), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=1, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling
    """
    

if __name__=="__main__":
    main()