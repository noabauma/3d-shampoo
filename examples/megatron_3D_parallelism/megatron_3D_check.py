import os

import time

import torch
import torch.distributed as dist

import deepspeed

from megatron import get_args
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron import mpu
from megatron.model import GPT2ModelPipe
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.fp16 import fp32_to_fp16
import numpy as np
import random

np.random.seed(42)
random.seed(42)
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

    if world_size > 1:
        assert dist.is_available()

        master_addr = os.environ.get('MASTER_ADDR', DEFAULT_MASTER_ADDR)
        master_port = os.environ.get('MASTER_PORT', DEFAULT_MASTER_PORT)
        init_method = 'tcp://' + master_addr + ':' + master_port       

        deepspeed.init_distributed(dist_backend=backend, verbose=True, init_method=init_method, rank=world_rank, world_size=world_size)

        assert dist.get_rank() == world_rank
        assert dist.get_world_size() == world_size

    return local_rank, local_size, world_rank, world_size

def get_batch_pipe(data):
    """A modification of get_batch() to work with the latest batch instead of an iterator. """
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    # unpack data
    if args.fp16:
        # cast to fp16 because pipeline parallelism skips the FP16 wrapper.
        return fp32_to_fp16((tokens, position_ids, attention_mask)), fp32_to_fp16((labels, loss_mask))
    else:
        return (tokens, position_ids, attention_mask), (labels, loss_mask)



def model_provider():
    """Build the model."""

    args = get_args()

    print_rank_0('building GPT2 model ...')
    if args.pipe_parallel_size == 0:
        model = GPT2Model(num_tokentypes=0, parallel_output=True)
        raise RuntimeError("We are only accepting args.pipe_parallel_size >= 1 atm for being consistent with our measurements?")
    else:
        model = GPT2ModelPipe(num_tokentypes=0, parallel_output=True, topology=mpu.get_topology())
        # This is a hack to give us a reference to get_batch_pipe from within training.py
        # We need to call model.set_batch_fn after deepspeed.initialize
        model._megatron_batch_fn = get_batch_pipe
        #model._megatron_batch_fn = None


    return model




if __name__ == "__main__":
    local_rank, local_size, world_rank, world_size = init_dist_process_group()

    node = os.environ.get('SLURMD_NODENAME', local_rank)

    #print(f"Hello I am node: {node}, with world_rank {world_rank}, local_rank {local_rank} and world_size {world_size}")

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron()

    args = get_args()
    timers = get_timers()

    #args.padded_vocab_size = 0

    args.shampoo = False #else torch.optim.SGD as control

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # set parameters of the model all to ones
    for p in model.parameters():
        if p.requires_grad:
            p.data = torch.ones_like(p)

    # warmup
    """
    for i in range(3):
        #print_rank_0(f"step:{i}")
        x = torch.randint(low=0, high=5000, size=(args.batch_size,args.seq_length))
        #print("x.shape: ", x.shape)
        _dict = {'text': x}
        batch = [_dict]*1
        
        loss = model.train_batch(data_iter=iter(batch))

    print_rank_0(f"done with learning loss: {loss}")
    """

    

    # training (throughput measure)
    train_time = 0
    n_iters = 1
    for i in range(n_iters):
        #x = torch.randint(low=0, high=5000, size=(args.batch_size,args.seq_length))
        x = torch.randint(low=1, high=2, size=(args.batch_size,args.seq_length))
        #print("x.shape: ", x.shape)
        _dict = {'text': x}
        batch = [_dict]   
        
        start = time.time()
        loss = model.train_batch(data_iter=iter(batch))
        train_time += time.time() - start

    avg_train_time = train_time/n_iters

    num_tokens = args.batch_size
    topology = model.topology()
    dp_dim = topology.get_dim('data')
    pp_dim = topology.get_dim('pipe')
    op_dim = topology.get_dim('model')

    print_rank_0(f"Throughput [tokens/s]: {num_tokens*dp_dim/avg_train_time}")
    print_rank_0(f"loss: {loss}")
    
    dist.barrier()

    # memory stats
    print_rank_0("CUDA memory stats: ")
    for i in range(world_size):
        if world_rank == i:
            print(f"GPU: {world_rank},{node} mem alloc [GB]: ",
                  torch.cuda.memory_allocated(device=local_rank)/1e9,
                  " max mem alloc [GB]: ",
                  torch.cuda.max_memory_allocated(device=local_rank)/1e9, 
                  flush=True)
        dist.barrier()


    write_files = True
    if write_files:
        if args.shampoo:
            # some file writing N_DP - N_PP - N_OP
            file_name = "output_files/" + str(dp_dim) + "-" + str(pp_dim) + "-" + str(op_dim) + "[" + str(world_rank) + "," + str(world_size) + "]"+".txt"

            print_rank_0(file_name)

            f = open(file_name, "w")

            if world_rank == 0: # extra information
                f.write(str(topology) + "\n")

            #f.write(str([p.shape for p in model.parameters() if p.requires_grad_]) + "\n")
            #f.write(str(optimizer.partitioned_modules) + "\n")
            try:
                partitioning = optimizer.partitioned_modules.tolist()
            except:
                partitioning = optimizer.partitioned_modules

            preconditioners = optimizer.precs

            f.write("name;module;shape;partitioning_ranks;precs\n")

            data_comm_lists = topology.get_axis_comm_lists('data') #creates suitable communicator groups along data parallelism
            for comm_list in data_comm_lists:
                if world_rank in comm_list:
                    comm_list_ = comm_list

            for name, module in model.named_modules():
                if hasattr(module, 'weight'):
                    if not module.weight.requires_grad:
                        raise RuntimeError(f"{world_rank}: there are layers which don't require grad :O")
                    
                    if world_rank == comm_list_[partitioning[0]]:
                        f.write(name + ";" + str(module) + ";" + str(tuple(module.weight.shape)) + ";" + str(partitioning.pop(0)) + ";" + str(preconditioners.pop(0)) + "\n")
                    else:
                        f.write(name + ";" + str(module) + ";" + str(tuple(module.weight.shape)) + ";" + str(partitioning.pop(0)) + ";" + "\n")

                if hasattr(module, 'bias'):
                    if not module.bias.requires_grad:
                        raise RuntimeError(f"{world_rank}: bias of this layer doesn't require grad :O")
                    
                    if world_rank == comm_list_[partitioning[0]]:
                        f.write(name + ";" + str(module) + ";" + str(tuple(module.bias.shape)) + ";" + str(partitioning.pop(0)) + ";" + str(preconditioners.pop(0)) + "\n")
                    else:
                        f.write(name + ";" + str(module) + ";" + str(tuple(module.bias.shape)) + ";" + str(partitioning.pop(0)) + ";" + "\n")

            f.close()
        else:
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            # some file writing N_DP - N_PP - N_OP
            file_name = "output_files/" + str(dp_dim) + "-" + str(pp_dim) + "-" + str(op_dim) + "[" + str(world_rank) + "," + str(world_size) + "]"+".txt"

            print_rank_0(file_name)

            f = open(file_name, "w")

            if world_rank == 0: # extra information
                f.write(str(topology) + "\n")

            #f.write(str([p.shape for p in model.parameters() if p.requires_grad_]) + "\n")
            #f.write(str(optimizer.partitioned_modules) + "\n")
            f.write("name;module;shape;parameters\n")

            data_comm_lists = topology.get_axis_comm_lists('data') #creates suitable communicator groups along data parallelism
            for comm_list in data_comm_lists:
                if world_rank in comm_list:
                    comm_list_ = comm_list

            for name, module in model.named_modules():
                if hasattr(module, 'weight'):
                    if not module.weight.requires_grad:
                        raise RuntimeError(f"{world_rank}: there are layers which don't require grad :O")

                    f.write(name + ";" + str(module) + ";" + str(tuple(module.weight.shape)) + ";" + str(module.weight.data) + ";" + "\n")

                if hasattr(module, 'bias'):
                    if not module.bias.requires_grad:
                        raise RuntimeError(f"{world_rank}: bias of this layer doesn't require grad :O")
                    
                    f.write(name + ";" + str(module) + ";" + str(tuple(module.bias.shape)) + ";" + str(module.bias.data) + ";" + "\n")

            f.close()
        