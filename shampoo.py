# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of Shampoo."""

from __future__ import print_function

import re # to find a number in a string
import enum
import itertools

from dataclasses import dataclass
import matrix_functions
import numpy as np
import torch
import torch.optim as optim

from torch.nn.utils import vector_to_parameters, parameters_to_vector

import torch.distributed as dist

# only used to predict comp cost for partitioning
class Fake_shape_class:
  def __init__(self, shape: tuple):
    self.shape = shape

  def shape(self):
    return self.shape


# Grafting is a technique to fix the layerwise scale of Shampoo optimizer.
# https://arxiv.org/pdf/2002.11803.pdf studies this in detail. This
# allows us to plugin the Shampoo optimizer into settings where SGD/AdaGrad
# is already well tuned. Grafting onto Shampoo means take the Shampoo direction,
# but use the step magnitude from the grafted optimizer such as Adagrad or SGD.
class LayerwiseGrafting(enum.IntEnum):
  NONE = 0
  SGD = 1
  ADAGRAD = 2


@dataclass
class ShampooHyperParams:
  """Shampoo hyper parameters."""
  beta2: float = 1.0
  diagonal_eps: float = 1e-6
  matrix_eps: float = 1e-12
  weight_decay: float = 0.0
  inverse_exponent_override: int = 0  # fixed exponent for preconditioner, if >0
  start_preconditioning_step: int = 1
  # Performance tuning params for controlling memory and compute requirements.
  # How often to compute preconditioner.
  preconditioning_compute_steps: int = 1
  # How often to compute statistics.
  statistics_compute_steps: int = 1
  # Block size for large layers (if > 0).
  # Block size = 1 ==> Adagrad (Don't do this, extremely inefficient!)
  # Block size should be as large as feasible under memory/time constraints.
  block_size: int = 0 #normal default: 128
  # Automatic shape interpretation (for eg: [4, 3, 1024, 512] would result in
  # 12 x [1024, 512] L and R statistics. Disabled by default which results in
  # Shampoo constructing statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
  best_effort_shape_interpretation: bool = False
  # to ignore certain type of layers to precondition like the embedding layer
  # [(name, module) for name, module in model.named_modules() if (hasattr(module, 'weight') or hasattr(module, 'bias'))]
  named_modules: list = None
  # If model parallelism is also altered as well, any given parallel configuration should have the same shapes of prec matrices
  gpt2_fair_blocks: bool = False
  # Type of grafting (SGD or AdaGrad).
  # https://arxiv.org/pdf/2002.11803.pdf
  graft_type: int = LayerwiseGrafting.SGD
  # Nesterov momentum
  nesterov: bool = True


class Graft:
  """Base class to perform grafting onto Shampoo. This class does no grafting.
  """

  def __init__(self, hps, unused_var):
    self.hps = hps

  def add_statistics(self, grad):
    pass

  def precondition_gradient(self, grad):
    return grad

  def update_momentum(self, update, unused_beta1):
    return update


class SGDGraft(Graft):
  """Graft using SGD+momentum.

  momentum maintains an exponentially weighted moving average of gradients.
  """

  def __init__(self, hps, var):
    super(SGDGraft, self).__init__(hps, var)
    self.momentum = torch.zeros_like(var.data, device=var.get_device())

  def update_momentum(self, update, beta1):
    self.momentum.mul_(beta1).add_(update)
    return self.momentum


class AdagradGraft(SGDGraft):
  """Graft using Adagrad.

  Essentially an implementation of Adagrad with momentum.
  """

  def __init__(self, hps, var):
    super(AdagradGraft, self).__init__(hps, var)
    self.statistics = torch.zeros_like(var.data, device=var.get_device())

  def add_statistics(self, grad):
    self.statistics.add_(grad * grad)

  def precondition_gradient(self, grad):
    return grad / (torch.sqrt(self.statistics) + self.hps.diagonal_eps)


class BlockPartitioner:
  """Partitions a tensor into smaller tensors for preconditioning.

    For example, if a variable has shape (4096, 512), we might split the
    4096 into 4 blocks, so we effectively have 4 variables of size
    (1024, 512) each.
  """

  def __init__(self, var, hps, enum):
    self._shape = var.shape
    self._splits = []
    self._split_sizes = []
    split_sizes = []
    # We split var into smaller blocks. Here we store the metadata to make
    # that split.

    if hps.gpt2_fair_blocks:
      module_name = hps.named_modules[enum][0]
      if ("query_key_value" in module_name) or ("dense_h_to_4h" in module_name):
        partition_dim = [True, False]
      elif ("attention.dense" in module_name) or ("dense_4h_to_h" in module_name):
        partition_dim = [False, True]
      else:
        partition_dim = [False]*len(var.shape)
      
    
    for i, d in enumerate(var.shape):
      if hps.gpt2_fair_blocks:
        if partition_dim[i] and not (hps.world_size == hps.mp_dim):   
          hps.block_size = d // (hps.world_size // hps.mp_dim)
          # d-1, otherwise split appends a 0-size array.
          nsplit = (d-1) // hps.block_size
          indices = (np.arange(nsplit, dtype=np.int32) + 1) * hps.block_size
          sizes = np.ones(nsplit + 1, dtype=np.int32) * hps.block_size
          sizes[-1] = d - indices[-1]
          self._splits.append((i, indices))
          self._split_sizes.append((i, sizes))
          split_sizes.append(sizes)
        else:
          split_sizes.append(np.array([d], dtype=np.int32))
      else:
        if hps.block_size > 0 and d > hps.block_size:
          # d-1, otherwise split appends a 0-size array.
          nsplit = (d-1) // hps.block_size
          indices = (np.arange(nsplit, dtype=np.int32) + 1) * hps.block_size
          sizes = np.ones(nsplit + 1, dtype=np.int32) * hps.block_size
          sizes[-1] = d - indices[-1]
          self._splits.append((i, indices))
          self._split_sizes.append((i, sizes))
          split_sizes.append(sizes)
        else:
          split_sizes.append(np.array([d], dtype=np.int32))
    self._num_splits = len(split_sizes)
    self._preconditioner_shapes = []
    for t in itertools.product(*split_sizes):
      self._preconditioner_shapes.extend([[d, d] for d in t])


  def shapes_for_preconditioners(self):
    return self._preconditioner_shapes

  def num_splits(self):
    return self._num_splits

  def partition(self, tensor):
    """Partition tensor into blocks."""

    assert tensor.shape == self._shape
    tensors = [tensor]
    for (i, sizes) in self._split_sizes:
      tensors_local = []
      for t in tensors:
        tensors_local.extend(
            torch.split(t, tuple(sizes), dim=i))
      tensors = tensors_local
    return tensors

  def merge_partitions(self, partitions):
    """Merge partitions back to original shape."""

    for (i, indices) in reversed(self._splits):
      n = len(indices) + 1
      partial_merged_tensors = []
      ind = 0
      while ind < len(partitions):
        partial_merged_tensors.append(
            torch.cat(partitions[ind:ind + n], axis=i))
        ind += n
      partitions = partial_merged_tensors
    assert len(partitions) == 1
    return partitions[0]


def _merge_small_dims(shape_to_merge, max_dim):
  """Merge small dimensions.

  If there are some small dimensions, we collapse them:
  e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
       [1, 2, 768, 1, 2048] --> [2, 768, 2048]

  Args:
    shape_to_merge: Shape to merge small dimensions.
    max_dim: Maximal dimension of output shape used in merging.

  Returns:
    Merged shape.
  """
  resulting_shape = []
  product = 1
  for d in shape_to_merge:
    if product * d <= max_dim:
      product *= d
    else:
      if product > 1:
        resulting_shape.append(product)
      product = d
  if product > 1:
    resulting_shape.append(product)
  return resulting_shape


class Preconditioner:
  """Compute statistics/shape from gradients for preconditioning."""

  def __init__(self, enum, var, hps):
    self._hps = hps
    self._original_shape = var.shape
    self._transformed_shape = var.shape
    if hps.best_effort_shape_interpretation:
      self._transformed_shape = _merge_small_dims(
          self._original_shape, hps.block_size)


    reshaped_var = torch.reshape(var, self._transformed_shape)
    self._partitioner = BlockPartitioner(reshaped_var, hps, enum)
    shapes = self._partitioner.shapes_for_preconditioners()
    rank = len(self._transformed_shape)
    device = var.get_device()
    if rank <= 1 or "embed" in self._hps.named_modules[enum][0]:
      self.statistics = []
      self.preconditioners = []
    else:
      eps = self._hps.matrix_eps
      self.statistics = [eps * torch.eye(s[0], device=device) for s in shapes]
      self.preconditioners = [torch.eye(s[0], device=device) for s in shapes]

  def add_statistics(self, grad):
    """Compute statistics from gradients and add to the correct state entries.

    Args:
      grad: Gradient to compute statistics from.
    """
    if not self.statistics: return
    reshaped_grad = torch.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    w1 = self._hps.beta2
    w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
    rank = len(self._transformed_shape)
    for j, grad in enumerate(partitioned_grads):
      for i in range(rank):
        axes = list(range(i)) + list(range(i + 1, rank))
        stat = torch.tensordot(grad, grad, [axes, axes])
        self.statistics[j*rank + i].mul_(w1).add_(stat, alpha=w2)

  def exponent_for_preconditioner(self):
    """Returns exponent to use for inverse-pth root M^{-1/p}."""
    if self._hps.inverse_exponent_override > 0:
      return self._hps.inverse_exponent_override
    return 2 * len(self._transformed_shape)

  def compute_preconditioners(self):
    """Compute L^{-1/exp} for each stats matrix L."""
    exp = self.exponent_for_preconditioner()
    eps = self._hps.matrix_eps
    for i, stat in enumerate(self.statistics):
      self.preconditioners[i] = matrix_functions.ComputePower(
          stat, exp, ridge_epsilon=eps, iter_count=20, fix_iter=True)

  def preconditioned_grad(self, grad):
    """Precondition the gradient.

    Args:
      grad: A gradient tensor to precondition.

    Returns:
      A preconditioned gradient.
    """
    if not self.preconditioners: return grad
    reshaped_grad = torch.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    preconditioned_partitioned_grads = []
    num_splits = self._partitioner.num_splits()
    for i, grad in enumerate(partitioned_grads):
      preconditioners_for_grad = self.preconditioners[i * num_splits:(i + 1) *
                                                      num_splits]
      rank = len(grad.shape)
      precond_grad = grad
      for j in range(rank):
        preconditioner = preconditioners_for_grad[j]
        precond_grad = torch.tensordot(
            precond_grad, preconditioner, [[0], [0]])
      preconditioned_partitioned_grads.append(precond_grad)
    merged_grad = self._partitioner.merge_partitions(
        preconditioned_partitioned_grads)
    return torch.reshape(merged_grad, self._original_shape)


STEP = 'step'
MOMENTUM = 'momentum'
PRECONDITIONER = 'preconditioner'
GRAFT = 'graft'


class Shampoo(optim.Optimizer):
  """The Shampoo optimizer."""

  def __init__(self,
               params,
               world_rank=0,
               world_size=1, 
               topology=None,                 # dictionary of the topology: e.g. {ProcessCoord(pipe=0, data=0): 0, ProcessCoord(pipe=0, data=1): 1, ProcessCoord(pipe=1, data=0): 2, ProcessCoord(pipe=1, data=1): 3}
               shapes=None,                   # list of tuples of all the shapes (needed for ZeRO stage >= 1 and for DP). Should be like this: [tuple(p.shape) for p in model.parameters() if p.requires_grad]
               zero_stage=0,                  # weither ZeRO optimization is active or not {0,1,2,3}
               partition_by_num_layers=False, # Sometimes, it is of interest to split by number of layers, instead of predicting the cost of each layer
               gpt2_partitioning=False,       # partition between the transformer encoder layers
               gpt2_nlayers=None,             # number of transformer encoder layers
               lr=1.0,
               momentum=0.9,
               hyperparams=ShampooHyperParams()):

    assert shapes is not None, "initialize shampoo with the given shapes!"
    assert len(shapes) == len(hyperparams.named_modules), f"they are not the same size {len(shapes)} != {len(hyperparams.named_modules)}!"
    self.shapes = shapes

    assert zero_stage == 0, "Shampoo does not work with ZeRO stage > 0, because ZeRO does not store the prec matrices!"
    self.zero_stage = zero_stage

    self.world_rank = world_rank
    self.world_size = world_size

    self.topology = topology

    # let's create process group for the data parallel parts!
    if topology is not None:
      if topology.get_dim('data') > 1: # only build dp_groups if data parallelism is active!
        self.data_comm_lists = topology.get_axis_comm_lists('data') #creates suitable communicator groups along data parallelism
        self.dp_groups = []
        for comm_list in self.data_comm_lists:
          if self.world_rank in comm_list:
            self.comm_list = comm_list
          self.dp_groups.append(dist.new_group(ranks=comm_list, backend='nccl'))
      else:
        self.dp_groups = None

      try:
        self.mp_dim = self.topology.get_dim('model')
      except:
        self.mp_dim = 1
        print("Shampoo initialize: No model parallelism, but ok.")

      try:
        self.pp_dim = self.topology.get_dim('pipe')
      except:
        self.pp_dim = 1
        print("Shampoo initialize: No pipeline parallelism, but ok.")
      
      try:
        self.dp_dim = self.topology.get_dim('data')
      except:
        self.dp_dim = 1
        print("Shampoo initialize: No data parallelism, but ok.")

    else:
      self.dp_groups = None
      self.dp_dim = 1
      self.pp_dim = 1
      self.mp_dim = 1
    

    defaults = dict(lr=lr, momentum=momentum)
    self.hps = hyperparams

    super(Shampoo, self).__init__(params, defaults)

    # fair block partitioning if gpt2
    if self.hps.gpt2_fair_blocks:
      assert self.hps.block_size == 0, f" block_size has to be zero in case of fair gpt2 blocks {self.hps.block_size}!"
      self.hps.world_size = self.world_size
      self.hps.mp_dim = self.mp_dim


    # lets make the partitioning
    self.gpt2_nlayers = gpt2_nlayers
    self.gpt2_partitioning = gpt2_partitioning
    self.partition_by_num_layers = partition_by_num_layers
    assert not (gpt2_partitioning and partition_by_num_layers), f"they can't be both true! {gpt2_partitioning}, {partition_by_num_layers}"

    if gpt2_partitioning:
      assert self.gpt2_nlayers is not None
      self.splits, self.partitioned_modules = self.get_distr_prec_partition_gpt2()
    else:
      self.splits, self.partitioned_modules = self.get_distr_prec_partition()

    """
    for i in range(world_size):
      if i == world_rank:
        print("@splits and partioning ", world_rank, ", ", self.splits, "\n", self.partitioned_modules, flush=True)
      dist.barrier()
    """

    """
    # no longer needed
    # lets make self.shape smaller according to the number of data parallelism if ZeRO_stage >= 1 is active
    if (self.zero_stage > 0) and (self.dp_groups is not None):
      dp_dim = self.topology.get_dim('data')
      total_numel = 0
      for shape in self.shapes:
        numel = 1
        for s in shape:
          numel *= s
        total_numel += numel

      assert total_numel % dp_dim == 0, f"Not even {total_numel} splittable by the {dp_dim}."

      subtotal_numel = total_numel // dp_dim
      
      sub_shapes = []
      sub_shape = []
      sub_numel = 0
      for shape in self.shapes:
        numel = 1
        for s in shape:
          numel *= s
        sub_numel += numel
        sub_shape.append(shape)
        # if the split is possible, add it to the sub_shapes list and restart until all shapes are done
        if sub_numel == subtotal_numel:
          sub_shapes.append(sub_shape)
          sub_shape = []
          sub_numel = 0
        elif sub_numel > subtotal_numel:
          raise RuntimeError(f"This model is not splittable by layers with the given DP dimension: {dp_dim}!")

      # let's find which GPU has which sub_shape list
      for comm_list in self.data_comm_lists:
        if self.world_rank in comm_list:
          idx = comm_list.index(self.world_rank)
          break

      self.shapes = sub_shapes[idx]
    """

        
  def get_distr_prec_partition_gpt2(self):
    assert len(self.shapes) == len(self.hps.named_modules)

    num_layers = len(self.shapes)
    if self.dp_dim == 1:
      return [], [0]*num_layers
    
    layers_reindexed = np.ones(num_layers, dtype=int)*-2 # embed layers = -1, ....
    for enum, p_shape in enumerate(self.shapes):
      if "embed" in self.hps.named_modules[enum][0]: # embedding layer
        layers_reindexed[enum] = -1
      elif len(re.findall(r'\d+', self.hps.named_modules[enum][0])) > 0:  # transformer layers
        layer_index = int(re.findall(r'\d+', self.hps.named_modules[enum][0])[0])
        if layer_index == (self.gpt2_nlayers + 3): # last layernorm
          layers_reindexed[enum] = layer_index - 4
        elif layer_index == 1:
          raise RuntimeError(f"We assumed indexing starts at '2'. This layer has index '1' re-do the code: {self.hps.named_modules[enum]}?")
        else:
          layers_reindexed[enum] = layer_index - 2

      else:
        raise RuntimeError(f"What is this layer: {self.hps.named_modules[enum]}?")
      
    assert -2 not in layers_reindexed, f"every layer should have an index: {layers_reindexed}"

    assert np.all(layers_reindexed[:-1] <= layers_reindexed[1:]), f"array is not sorted: {layers_reindexed}"

    for index in layers_reindexed:
      if index >= 0:
        first_layer = index
        break

    layers_reindexed[layers_reindexed == -1] = first_layer

    min_index = min(layers_reindexed)

    assert min_index >= 0, f"something went wrong: {layers_reindexed}"

    assert (self.gpt2_nlayers % self.dp_dim) == 0 and (self.gpt2_nlayers % self.dp_dim) == 0, f"gpt2_nlayers: {self.gpt2_nlayers} has to be divisible by dp_dim: {self.dp_dim} and by pp_dim: {self.pp_dim}!"

    layers_reindexed -= min_index
    layers_reindexed //= int(self.gpt2_nlayers/(self.dp_dim*self.pp_dim))

    split_list = []
    same_index = layers_reindexed[0]
    cnt = 1
    for i in range(1,num_layers):
      if same_index == layers_reindexed[i]:
        cnt += 1
      else:
        same_index = layers_reindexed[i]
        shift = split_list[-1] if len(split_list) > 0 else 0
        split_list.append(cnt + shift)
        cnt = 1

    assert len(np.unique(layers_reindexed)) - 1 == len(split_list), f"they do not match {layers_reindexed} {split_list}"

    return split_list, layers_reindexed

    






  def get_distr_prec_partition(self):
    """
    Distributes the workload by computational cost of each layer for total number of Data parallelism
    TODO: multiple GPUs for on layer
    e.g.
    1 GPU for ResNet18:
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    3 GPUs for ResNet18:
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2]
    8 GPUs for ResNet18:
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 6, 7]
    21 or more GPUs for ResNet18:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    2 GPUs for 3 layers MLP (if first layer is bigger than 2nd and 3rd):
    [0,1,1]
    """

    total_comp_cost = 0
    comp_cost_layers = []
    shapes_list = []
    for enum, p_shape in enumerate(self.shapes):
      if self.partition_by_num_layers:
        comp_cost = 1
        total_comp_cost += comp_cost
        comp_cost_layers.append(comp_cost) 
      else:
        _transformed_shape = _merge_small_dims(p_shape, self.hps.block_size)
        _transformed_shape_class = Fake_shape_class(_transformed_shape)
        _partitioner = BlockPartitioner(_transformed_shape_class, self.hps)
        shapes = _partitioner.shapes_for_preconditioners()

        #shapes_list.append(_transformed_shape) # only for debugging
        comp_cost = self.computational_cost(shapes)
        total_comp_cost += comp_cost
        comp_cost_layers.append(comp_cost)

    num_layers = len(comp_cost_layers)

    partitions = [0]*num_layers
    if self.topology is None or self.dp_groups is None or self.dp_dim == 1:
      return [], partitions
    elif num_layers > self.dp_dim:
      split_list = np.array([0])

      for rank in range(self.dp_dim-1):
        if rank == 0:
          split_list = np.append(split_list, self.next_split(comp_cost_layers))
        else:
          sub_sums = []
          for i in range(1, len(split_list)):
            
            local_comp_cost = np.sum(comp_cost_layers[split_list[i-1]:split_list[i]])
            sub_sums.append(local_comp_cost)
            
            if i == len(split_list) - 1:
              local_comp_cost = np.sum(comp_cost_layers[split_list[i]:])
              sub_sums.append(local_comp_cost)

          while(True):
            i = np.argmax(sub_sums)
            if i == len(sub_sums) - 1:
                sub_comp_cost_layers = comp_cost_layers[split_list[i]:]
                shift = split_list[i]
            else:
                sub_comp_cost_layers = comp_cost_layers[split_list[i]:split_list[i+1]]
                shift = split_list[i]

            if len(sub_comp_cost_layers) > 1:
                break
            else:
                sub_sums[i] = -1


          split_list = np.append(split_list, self.next_split(sub_comp_cost_layers) + shift)
          split_list = np.sort(split_list)

      sub_sums = []
      for i in range(1, len(split_list)):
          
        local_comp_cost = np.sum(comp_cost_layers[split_list[i-1]:split_list[i]])
        sub_sums.append(local_comp_cost)
        
        if i == len(split_list) - 1:
          local_comp_cost = np.sum(comp_cost_layers[split_list[i]:])
          sub_sums.append(local_comp_cost)

      next_split = split_list[1]
      rank = 0
      for i in range(len(partitions)):
        if i == next_split:
          rank += 1
          if rank != self.topology.get_dim('data') - 1:
            next_split = split_list[rank+1]
          
        partitions[i] = rank
      return split_list[1:], partitions
    else: #atm, we do not support multiple gpus for one layer
      rank = 0
      for i in range(num_layers):
        partitions[i] = i
          
      return partitions[1:], partitions

  def computational_cost(self, shapes):
    """
    input: shape: [[x, x],[y, y],...] (Blockpartitioner.kronecker_factor_shape)
    output: returns the compuational cost of this Blockpartitioned layers
    """
    tmp_cost = 0
    for shape in shapes:
      assert len(shape) == 2
      assert shape[0] == shape[1]

      tmp_cost += shape[0]**0.4 # ATM simple O(n^3) assumption (maybe even less 0.4)

    return tmp_cost

  def next_split(self, subset_partitions):
    """
    deciding where the next split is happening
    
    input: subset_partitions: [] is a subset of comp_cost_layers
    output: index where to split (int)
    """
    assert len(subset_partitions) > 1

    x = np.array(subset_partitions)
    y = np.sum(subset_partitions)/2

    split_loc = len(x[np.cumsum(x) < y])

    #if split_loc == 0:  # for resnet and densenet, this is really good
    split_loc += 1
    
    return split_loc


  def init_var_state(self, enum, var, state):
    """Initialize the PyTorch state of for a single variable."""
    state[STEP] = 0
    state[MOMENTUM] = torch.zeros_like(var.data, device=var.get_device())
    state[PRECONDITIONER] = Preconditioner(enum, var, self.hps)
    if self.hps.graft_type == LayerwiseGrafting.ADAGRAD:
      state[GRAFT] = AdagradGraft(self.hps, var)
    elif self.hps.graft_type == LayerwiseGrafting.SGD:
      state[GRAFT] = SGDGraft(self.hps, var)
    else:
      state[GRAFT] = Graft(self.hps, var)

  def step(self, closure=None):
    hps = self.hps
    for group in self.param_groups:
      lr = group['lr']
      #print("shapes: ", [p.grad for p in group['params']], flush=True)
      if len(group['params']) == 1 and group['params'][0].ndim == 1: # when ZeRO stage >=1 is active

        assert self.zero_stage > 0
        
        vec = group['params'][0]
        device = vec.get_device()
        params = []
        grads = []
        check_numel = 0
        for shape in self.shapes:
          params.append(torch.empty(shape, dtype=torch.float32).to(device))
          grads.append(torch.empty(shape, dtype=torch.float32).to(device))

          numel = 1
          for s in shape:
            numel *= s
          check_numel += numel

        assert vec.numel() == check_numel, f"Not identical number of elements of the packed tensor: {vec.numel()} and number of elements of shapes: {check_numel}, shapes: {self.shapes}"

        vector_to_parameters(vec.grad, grads)
        vector_to_parameters(vec, params)

        torch.cuda.empty_cache()

        for p, grad in zip(params, grads):
          if grad is None: continue
          if grad.is_sparse:
            raise RuntimeError('Shampoo does not support sparse yet')
          state = self.state[p]
          if not state:
            self.init_var_state(p, state)
          state[STEP] += 1

          preconditioner = state[PRECONDITIONER]
          graft = state[GRAFT]

          # Gather statistics, compute preconditioners
          graft.add_statistics(grad)
          if state[STEP] % hps.statistics_compute_steps == 0:
            preconditioner.add_statistics(grad)
          if state[STEP] % hps.preconditioning_compute_steps == 0:
            preconditioner.compute_preconditioners()

          # Precondition gradients
          graft_grad = graft.precondition_gradient(grad)
          shampoo_grad = grad
          if state[STEP] >= self.hps.start_preconditioning_step:
            shampoo_grad = preconditioner.preconditioned_grad(grad)

          # Grafting
          graft_norm = torch.norm(graft_grad)
          shampoo_norm = torch.norm(shampoo_grad)
          shampoo_grad.mul_(graft_norm / (shampoo_norm + 1e-16))

          # Weight decay
          if self.hps.weight_decay != 0.0:
            shampoo_grad.add_(p.data, alpha=self.hps.weight_decay)
            graft_grad.add_(p.data, alpha=self.hps.weight_decay)

          # Momentum and Nesterov momentum, if needed
          state[MOMENTUM].mul_(group['momentum']).add_(shampoo_grad)
          graft_momentum = graft.update_momentum(grad, group['momentum'])

          if state[STEP] >= self.hps.start_preconditioning_step:
            momentum_update = state[MOMENTUM]
            wd_update = shampoo_grad
          else:
            momentum_update = graft_momentum
            wd_update = graft_grad

          if hps.nesterov:
            momentum_update.mul_(group['momentum']).add_(wd_update)

          # Final update
          p.data.add_(momentum_update, alpha=-lr)

        #vec.grads = parameters_to_vector(grads)
        #vec       = parameters_to_vector(params)

        torch.cuda.empty_cache()

      else:
        self.precs = []
        assert (self.dp_groups is None) or (len(group['params']) == len(self.partitioned_modules))
        for enum, p in enumerate(group['params']):
          if (self.dp_groups is None) or (self.world_rank == self.comm_list[self.partitioned_modules[enum]]):
            if p.grad is None: continue
            grad = p.grad.data
            if grad.is_sparse:
              raise RuntimeError('Shampoo does not support sparse yet')
            state = self.state[p]
            if not state:
              self.init_var_state(enum, p, state)
            state[STEP] += 1

            preconditioner = state[PRECONDITIONER]
            graft = state[GRAFT]

            # Gather statistics, compute preconditioners
            graft.add_statistics(grad)
            if state[STEP] % hps.statistics_compute_steps == 0:
              preconditioner.add_statistics(grad)
            if state[STEP] % hps.preconditioning_compute_steps == 0:
              preconditioner.compute_preconditioners()

            # Precondition gradients
            graft_grad = graft.precondition_gradient(grad)
            shampoo_grad = grad
            if state[STEP] >= self.hps.start_preconditioning_step:
              shampoo_grad = preconditioner.preconditioned_grad(grad)

            # Grafting
            graft_norm = torch.norm(graft_grad)
            shampoo_norm = torch.norm(shampoo_grad)
            shampoo_grad.mul_(graft_norm / (shampoo_norm + 1e-16))

            # Weight decay
            if self.hps.weight_decay != 0.0:
              shampoo_grad.add_(p.data, alpha=self.hps.weight_decay)
              graft_grad.add_(p.data, alpha=self.hps.weight_decay)

            # Momentum and Nesterov momentum, if needed
            state[MOMENTUM].mul_(group['momentum']).add_(shampoo_grad)
            graft_momentum = graft.update_momentum(grad, group['momentum'])

            if state[STEP] >= self.hps.start_preconditioning_step:
              momentum_update = state[MOMENTUM]
              wd_update = shampoo_grad
            else:
              momentum_update = graft_momentum
              wd_update = graft_grad

            if hps.nesterov:
              momentum_update.mul_(group['momentum']).add_(wd_update)

            # Final update
            p.data.add_(momentum_update, alpha=-lr)

            self.precs.append([prec.shape for prec in preconditioner.preconditioners])
            #precs.append([prec.shape for prec in preconditioner.preconditioners])
            #print("preconditioners: ", [prec.shape for prec in preconditioner.preconditioners])
            #print("grads: ", [p.grad for p in group['params']])
            """
            if self.mp_dim > 1:
              for i in range(self.world_size):
                if i == self.world_rank:
                  print("preconditioners: ", [(prec.shape, prec) for prec in preconditioner.preconditioners], flush=True)
                  dist.barrier()
            """

        # broadcast those parameters back to the other DP group if it exists (only needed for ZeRO_stage == 0)
        if self.dp_groups is not None:
          params = [p for p in group['params'] if p.grad is not None]

          params_list = []
          tensor_list = []
          for i in range(len(self.splits)):
            if i == 0:
              params_split = params[:self.splits[i]]
              params_list.append(params_split)
              tensor_list.append(parameters_to_vector(params_split))
            elif len(self.splits) > 1:
              params_split = params[self.splits[i-1]:self.splits[i]]
              params_list.append(params_split)
              tensor_list.append(parameters_to_vector(params_split))
            
            if i == len(self.splits) - 1:
              params_split = params[self.splits[i]:]
              params_list.append(params_split)
              tensor_list.append(parameters_to_vector(params_split))

          assert len(self.splits)+1 == len(tensor_list) <= self.dp_dim, str(self.splits) + ', ' + str(len(tensor_list)) + ', '  + str(self.topology.get_dim('data'))

          for enum, comm_list in enumerate(self.data_comm_lists):
            assert len(comm_list) == len(tensor_list)
            if self.world_rank in comm_list:
              handler_list = []
              for i in range(len(tensor_list)):
                handler = dist.broadcast(tensor_list[i], comm_list[i], group=self.dp_groups[enum], async_op=True) # maybe don't async? It's dangerous for multiple groups in nccl
                # TODO: maybe all_gather if gpt2
                
                handler_list.append(handler)

                for handler in handler_list:
                  handler.wait()

          for i in range(len(tensor_list)): # all GPUs unpack the new gotten params
            vector_to_parameters(tensor_list[i], params_list[i])

        
        """
        for i in range(self.world_size):
          if i == self.world_rank:
            print("@topology: ", self.topology, [p.shape for p in group['params']], precs, self.partitioned_modules, self.world_rank, self.world_size, self.hps.ignore_embedding_layer, flush=True)
          dist.barrier()
        """
        
        
