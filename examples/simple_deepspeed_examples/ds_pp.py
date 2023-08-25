import os
import argparse
from collections import OrderedDict

import torch
import torch.distributed as dist


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe.schedule import *

from deepspeed.profiling.flops_profiler import FlopsProfiler

# loading shampoo optimizer
import sys
sys.path.append('../shampoo_optimizer/')
import shampoo

torch.manual_seed(42)

DEFAULT_MASTER_ADDR = '127.0.0.1'
DEFAULT_MASTER_PORT = '1234'

def init_dist_process_group(backend='nccl', is_high_priority=True):
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

        os.environ['RANK'] = str(world_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(0)
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1
        local_size = 1

    if world_size > 0:
        assert dist.is_available()

        master_addr = os.environ.get('MASTER_ADDR', DEFAULT_MASTER_ADDR)
        master_port = os.environ.get('MASTER_PORT', DEFAULT_MASTER_PORT)
        init_method = 'tcp://' + master_addr + ':' + master_port       

        deepspeed.init_distributed(dist_backend=backend, verbose=True, init_method=init_method, rank=world_rank, world_size=world_size) #this method needed on Piz Daint
        #deepspeed.init_distributed(dist_backend=backend, verbose=True)

        assert dist.get_rank() == world_rank
        assert dist.get_world_size() == world_size

    return local_rank, local_size, world_rank, world_size


def main():
    local_rank, local_size, world_rank, world_size = init_dist_process_group()

    # only needed for Barry (multi GPU on one node)
    if local_size == world_size:
        torch.cuda.set_device(local_rank)

    node = os.environ.get('SLURMD_NODENAME', local_rank)

    #print(f"Hello I am node: {node}, with world_rank {world_rank}, local_rank {local_rank} and world_size {world_size}")

    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=local_rank,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    batchsize = 1

    hidden_dim = 2
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('fc1', nn.Linear(hidden_dim, hidden_dim, bias=False)),
        ('relu1', nn.LeakyReLU()),
        ('fc2', nn.Linear(hidden_dim, hidden_dim, bias=False)),
        ('relu2', nn.LeakyReLU()),
        ('fc3', nn.Linear(hidden_dim, hidden_dim, bias=False)),
        ('relu3', nn.LeakyReLU()),
        ('fc4', nn.Linear(hidden_dim, hidden_dim, bias=False)),
    ]))
    
    model = PipelineModule(layers=model, loss_fn=F.mse_loss, num_stages=1)

    """
    optimizer = shampoo.Shampoo(params=model.parameters(),
                                world_rank=world_rank,
                                world_size=world_size,
                                topology=model.topology(), 
                                shapes=[tuple(p.shape) for p in model.parameters() if p.requires_grad], 
                                lr=1e-1, 
                                momentum=0.9, 
                                hyperparams=shampoo.ShampooHyperParams(ignore_embedding_layer=True))
    """
    #optimizer = shampoo.Shampoo(params=model.parameters(), topology=model.topology(), shapes=[tuple(p.shape) for p in model.parameters() if p.requires_grad] ,lr=1e-1, momentum=0.9, hyperparams=shampoo.ShampooHyperParams())
    #optimizer = deepspeed.ops.adam.FusedAdam(model.parameters(), lr=1e-1)
    optimizer = optim.SGD(params=model.parameters(), lr=1e-1)
    
    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=model,
                                                         optimizer=optimizer
                                                        )

    prof = FlopsProfiler(model_engine)
    profile = True


    for step in range(1):  #two epochs for nsys
        x = torch.rand(batchsize,2,hidden_dim//2) + world_rank
        t = torch.rand(batchsize, hidden_dim)
        batch = [(x, t)]

        #optimizer.zero_grad()
        #print("data before:\n", [p.data for p in model_engine.parameters()], "\n")
        
        if profile:
            prof.start_profile()
            #forward() & backward() method
            loss = model_engine.train_batch(data_iter=iter(batch)) #Pipeline method
            #loss = model_engine.eval_batch(data_iter=iter(batch))

            flops = prof.get_total_flops(as_string=True)
            params = prof.get_total_params(as_string=True)
            prof.print_model_profile()
            prof.end_profile()

            print("flops: ", flops)
            print("params: ", params)
        else:
            #forward() & backward() method
            loss = model_engine.train_batch(data_iter=iter(batch)) #Pipeline method

        #print("data after:\n", [p.data for p in model_engine.parameters()], "\n")

    topology = model.topology()

    if world_rank == 0:
        print("CUDA memory stats: ", flush=True)
    for i in range(world_size):
        if world_rank == i:
            print(f"GPU{node} mem alloc [GB]: ",
                  torch.cuda.memory_allocated(device=local_rank)/1e9,
                  " max mem alloc [GB]: ",
                  torch.cuda.max_memory_allocated(device=local_rank)/1e9,
                  " model topology: ",
                  model.topology(), flush=True)
            print(topology.get_axis_comm_lists('pipe'))
            print(topology.get_axis_comm_lists('data'))
        dist.barrier()
    
    

if __name__=="__main__":
    main()