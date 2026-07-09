"""Smoke and correctness tests for 3D-Shampoo.

Run with:  python tests/test_shampoo_3d.py

Covers:
  * matrix_functions.ComputePower against a direct eigendecomposition
  * single-process Shampoo (topology=None) on CPU and GPU
  * block-partitioned and merged-shape preconditioning
  * skipping of embedding layers via named_modules
  * distributed preconditioning (data parallelism) with 2 and 3 ranks,
    using the gloo backend and a real DeepSpeed topology, checked for
    cross-rank consistency and parity with a single-process reference run
  * an emulated 3D-parallel run (2 pipeline stages x 2 model-parallel
    slices x 2 data-parallel replicas = 8 ranks) with a real DeepSpeed
    PipeModelDataParallelTopology, verifying that the preconditioning is
    split within each data-parallel group and nowhere else

The distributed tests only need CPUs, so everything here runs on a single
machine without any GPU requirement (GPU test is skipped if unavailable).
"""

import os
import sys
import socket
import tempfile
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.utils import parameters_to_vector

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import shampoo_3d
import matrix_functions

NSTEPS = 20
LR = 0.1


def make_model(seed, sizes, bias=True):
  torch.manual_seed(seed)
  layers = OrderedDict()
  for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
    layers[f'fc{i}'] = nn.Linear(n_in, n_out, bias=bias)
    if i < len(sizes) - 2:
      layers[f'relu{i}'] = nn.Tanh()
  return nn.Sequential(layers)


def batch_for_step(step, batchsize, n_in, n_out, device='cpu', seed=10_000):
  """Deterministic toy regression batch, identical on every rank.

  The batch is fixed (independent of the step) so that the loss has to
  shrink towards zero when the optimizer works.
  """
  g = torch.Generator().manual_seed(seed)
  x = torch.randn(batchsize, n_in, generator=g).to(device)
  t = torch.randn(batchsize, n_out, generator=g).to(device)
  return x, t


def train_step(model, optimizer, step, sizes, device='cpu', batch_seed=10_000):
  x, t = batch_for_step(step, 8, sizes[0], sizes[-1], device, seed=batch_seed)
  loss = F.mse_loss(model(x), t)
  model.zero_grad()
  loss.backward()
  optimizer.step()
  return loss.item()


def run_single_process(sizes, bias=True, device='cpu', hyperparams=None, nsteps=NSTEPS,
                       model_seed=42, batch_seed=10_000):
  model = make_model(model_seed, sizes, bias).to(device)
  optimizer = shampoo_3d.Shampoo_3D(model.parameters(), lr=LR, momentum=0.9,
                                    hyperparams=hyperparams)
  losses = [train_step(model, optimizer, s, sizes, device, batch_seed=batch_seed)
            for s in range(nsteps)]
  return losses, parameters_to_vector(model.parameters()).detach().cpu()


def check_loss_decreased(losses, name):
  first, last = losses[0], losses[-1]
  assert last < 0.5 * first, f"{name}: loss did not decrease enough ({first:.4f} -> {last:.4f})"
  print(f"  {name}: loss {first:.4f} -> {last:.4f}  OK")


def test_compute_power():
  torch.manual_seed(0)
  m = torch.randn(32, 64, dtype=torch.float64)
  a = m @ m.t() / 64 + 0.1 * torch.eye(32, dtype=torch.float64)
  a_before = a.clone()
  for p in (2, 4):
    root = matrix_functions.ComputePower(a, p, iter_count=100, ridge_epsilon=1e-12)
    evals, evecs = torch.linalg.eigh(a)
    expected = evecs @ torch.diag(evals.pow(-1.0 / p)) @ evecs.t()
    rel_err = torch.norm(root - expected) / torch.norm(expected)
    assert rel_err < 1e-4, f"ComputePower(p={p}) rel err {rel_err:.2e}"
    print(f"  ComputePower p={p}: rel err {rel_err:.2e}  OK")
  assert torch.equal(a, a_before), "ComputePower must not modify its input matrix"
  print("  ComputePower leaves the statistics matrix untouched  OK")


def test_single_process_cpu():
  losses, _ = run_single_process([16, 32, 8, 4])
  check_loss_decreased(losses, "plain MLP (CPU)")

  hps = shampoo_3d.ShampooHyperParams(block_size=8)
  losses, _ = run_single_process([16, 32, 8, 4], hyperparams=hps)
  check_loss_decreased(losses, "block_size=8")

  hps = shampoo_3d.ShampooHyperParams(graft_type=shampoo_3d.LayerwiseGrafting.ADAGRAD)
  losses, _ = run_single_process([16, 32, 8, 4], hyperparams=hps)
  check_loss_decreased(losses, "Adagrad grafting")


def test_merged_shapes():
  """A rank-3 weight gets merged to rank 2 with best_effort_shape_interpretation."""
  torch.manual_seed(42)
  model = nn.Sequential(nn.Conv1d(4, 8, 3, padding=1), nn.Flatten(), nn.Linear(8 * 6, 4))
  hps = shampoo_3d.ShampooHyperParams(best_effort_shape_interpretation=True, block_size=16)
  optimizer = shampoo_3d.Shampoo_3D(model.parameters(), lr=LR, momentum=0.9, hyperparams=hps)
  losses = []
  g = torch.Generator().manual_seed(20_000)
  x = torch.randn(8, 4, 6, generator=g)
  t = torch.randn(8, 4, generator=g)
  for step in range(NSTEPS):
    loss = F.mse_loss(model(x), t)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
  conv_weight = model[0].weight
  n_stats = len(optimizer.state[conv_weight]['preconditioner'].statistics)
  assert n_stats == 2, f"expected (8,4*3) merge -> 2 statistics, got {n_stats}"
  check_loss_decreased(losses, "conv with merged shapes")


def test_embedding_skip():
  torch.manual_seed(42)
  emb = nn.Embedding(10, 8)
  fc = nn.Linear(8, 2)
  model = nn.Sequential(emb, nn.Flatten(start_dim=1, end_dim=-1))

  named_modules = [('embedding', emb), ('fc', fc), ('fc', fc)]
  params = list(emb.parameters()) + list(fc.parameters())
  hps = shampoo_3d.ShampooHyperParams(named_modules=named_modules)
  optimizer = shampoo_3d.Shampoo_3D(params, lr=LR, momentum=0.9, hyperparams=hps)

  for step in range(3):
    g = torch.Generator().manual_seed(30_000 + step)
    idx = torch.randint(0, 10, (8,), generator=g)
    t = torch.randn(8, 2, generator=g)
    loss = F.mse_loss(fc(emb(idx)), t)
    emb.zero_grad(); fc.zero_grad()
    loss.backward()
    optimizer.step()

  assert optimizer.state[emb.weight]['preconditioner'].preconditioners == [], \
      "embedding layer must not be preconditioned"
  assert len(optimizer.state[fc.weight]['preconditioner'].preconditioners) == 2, \
      "linear layer must be preconditioned"
  print("  embedding layer skipped, linear layer preconditioned  OK")


def test_single_process_gpu():
  if not torch.cuda.is_available():
    print("  no GPU available, skipped")
    return
  losses, _ = run_single_process([16, 32, 8, 4], device='cuda')
  check_loss_decreased(losses, "plain MLP (GPU)")
  hps = shampoo_3d.ShampooHyperParams(block_size=8)
  losses, _ = run_single_process([16, 32, 8, 4], device='cuda', hyperparams=hps)
  check_loss_decreased(losses, "block_size=8 (GPU)")


def _dp_worker(rank, world_size, port, sizes, bias, partition_by_num_layers, result_file):
  """One data-parallel rank: same model and same data on every rank, so the
  gradients are identical (as they would be after DeepSpeed's allreduce) and
  only the preconditioning work is distributed."""
  import torch.distributed as dist
  dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                          rank=rank, world_size=world_size)
  from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
  topology = PipeDataParallelTopology(num_pp=1, num_dp=world_size)

  model = make_model(42, sizes, bias)
  optimizer = shampoo_3d.Shampoo_3D(model.parameters(),
                                    world_rank=rank,
                                    world_size=world_size,
                                    topology=topology,
                                    partition_by_num_layers=partition_by_num_layers,
                                    lr=LR, momentum=0.9)
  losses = []
  for step in range(NSTEPS):
    losses.append(train_step(model, optimizer, step, sizes))
    # after each step all ranks must hold bitwise identical parameters
    vec = parameters_to_vector(model.parameters()).detach()
    vecs = [torch.zeros_like(vec) for _ in range(world_size)]
    dist.all_gather(vecs, vec)
    for other in vecs[1:]:
      assert torch.equal(other, vecs[0]), \
          f"rank {rank}, step {step}: parameters differ across ranks"

  if rank == 0:
    torch.save({'losses': losses,
                'params': parameters_to_vector(model.parameters()).detach(),
                'partitioned_modules': list(optimizer.partitioned_modules),
                'splits': list(optimizer.splits)}, result_file)
  dist.barrier()
  dist.destroy_process_group()


def _free_port():
  with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    return s.getsockname()[1]


def run_dp_test(world_size, sizes, bias=True, partition_by_num_layers=False):
  result_file = os.path.join(tempfile.mkdtemp(), 'dp_result.pt')
  mp.spawn(_dp_worker,
           args=(world_size, _free_port(), sizes, bias, partition_by_num_layers, result_file),
           nprocs=world_size, join=True)
  result = torch.load(result_file, weights_only=False)

  ref_losses, ref_params = run_single_process(sizes, bias=bias)
  torch.testing.assert_close(result['params'], ref_params, rtol=1e-4, atol=1e-5)
  assert abs(result['losses'][-1] - ref_losses[-1]) < 1e-4 * max(1.0, abs(ref_losses[-1]))
  name = (f"DP={world_size}, {len(result['partitioned_modules'])} params, "
          f"owners {result['partitioned_modules']}")
  check_loss_decreased(result['losses'], name)
  print(f"    matches the single-process reference  OK")


def test_data_parallel():
  # cost-based partitioning, more parameters than ranks
  run_dp_test(2, [16, 32, 8, 4])
  # partitioning by number of layers
  run_dp_test(2, [16, 32, 8, 4], partition_by_num_layers=True)
  # three ranks, unbalanced layers
  run_dp_test(3, [16, 64, 8, 4])
  # fewer parameters than ranks (one rank idles during preconditioning)
  run_dp_test(2, [16, 4], bias=False)


def _shard_sizes(pipe, model):
  """The MLP a given (pipeline stage, model-parallel slice) coordinate holds.

  The pipeline stage decides which layers exist, the model-parallel slice
  decides their width (as a Megatron-style column split would); the widths
  differ on purpose so no two shards are alike."""
  hidden = 32 if model == 0 else 16
  return [16, hidden, 12, 8] if pipe == 0 else [8, hidden, 6, 4]


def _shard_seeds(pipe, model):
  return 42 + 10 * pipe + model, 10_000 + 100 * pipe + model


def _3d_worker(rank, world_size, port, result_dir):
  """One rank of an emulated 3D-parallel run on CPU: 2 pipeline stages x
  2 model-parallel slices x 2 data-parallel replicas = 8 ranks.

  A real 3D run gives each rank only its own shard of the model, and
  data-parallel replicas hold identical copies with identical (allreduced)
  gradients.  The optimizer never sees more than that shard plus the
  topology, so training an independent small MLP per (pipe, model)
  coordinate exercises exactly the code path of a real 3D run: reading the
  topology dims, building the comm groups along the data axis, partitioning
  the preconditioning work inside each group, and broadcasting the updates.
  """
  import torch.distributed as dist
  dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                          rank=rank, world_size=world_size)
  from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
  topology = PipeModelDataParallelTopology(num_pp=2, num_mp=2, num_dp=2)
  coord = topology.get_coord(rank)

  sizes = _shard_sizes(coord.pipe, coord.model)
  model_seed, batch_seed = _shard_seeds(coord.pipe, coord.model)
  model = make_model(model_seed, sizes)
  optimizer = shampoo_3d.Shampoo_3D(model.parameters(),
                                    world_rank=rank, world_size=world_size,
                                    topology=topology, lr=LR, momentum=0.9)

  # this rank's DP group partner must share its (pipe, model) coordinates
  assert rank in optimizer.comm_list and len(optimizer.comm_list) == 2
  partner = next(r for r in optimizer.comm_list if r != rank)
  pcoord = topology.get_coord(partner)
  assert (pcoord.pipe, pcoord.model) == (coord.pipe, coord.model), \
      f"rank {rank}: DP group mixes different model shards"

  losses = []
  for step in range(NSTEPS):
    losses.append(train_step(model, optimizer, step, sizes, batch_seed=batch_seed))
    # parameters must stay bitwise identical inside the DP group
    # (other (pipe, model) shards are unrelated models)
    vec = parameters_to_vector(model.parameters()).detach()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, (coord.pipe, coord.model, vec))
    for other_pipe, other_model, other_vec in gathered:
      if (other_pipe, other_model) == (coord.pipe, coord.model):
        assert torch.equal(other_vec, vec), \
            f"rank {rank}, step {step}: parameters differ inside the DP group"

  # the preconditioning must really be split: the two ranks of the DP group
  # own disjoint, non-empty parameter subsets that together cover the shard
  n_params = len(list(model.parameters()))
  owned = [i for i in range(n_params)
           if optimizer.comm_list[optimizer.partitioned_modules[i]] == rank]
  assert len(optimizer.precs) == len(owned), \
      f"rank {rank}: computed preconditioners for parameters it does not own"
  gathered = [None] * world_size
  dist.all_gather_object(gathered, (rank, owned))
  owned_by_rank = dict(gathered)
  assert owned_by_rank[rank] and owned_by_rank[partner], \
      f"DP group of rank {rank}: one rank does all the preconditioning"
  assert sorted(owned_by_rank[rank] + owned_by_rank[partner]) == list(range(n_params)), \
      f"DP group of rank {rank}: preconditioning does not cover the shard exactly once"

  if coord.data == 0:
    torch.save({'losses': losses,
                'params': parameters_to_vector(model.parameters()).detach(),
                'owned': (owned_by_rank[rank], owned_by_rank[partner])},
               os.path.join(result_dir, f'shard_p{coord.pipe}_m{coord.model}.pt'))
  dist.barrier()
  dist.destroy_process_group()


def test_3d_topology():
  world_size = 8
  result_dir = tempfile.mkdtemp()
  mp.spawn(_3d_worker, args=(world_size, _free_port(), result_dir),
           nprocs=world_size, join=True)
  for pipe in range(2):
    for model in range(2):
      result = torch.load(os.path.join(result_dir, f'shard_p{pipe}_m{model}.pt'),
                          weights_only=False)
      model_seed, batch_seed = _shard_seeds(pipe, model)
      ref_losses, ref_params = run_single_process(_shard_sizes(pipe, model),
                                                  model_seed=model_seed,
                                                  batch_seed=batch_seed)
      torch.testing.assert_close(result['params'], ref_params, rtol=1e-4, atol=1e-5)
      assert abs(result['losses'][-1] - ref_losses[-1]) < 1e-4 * max(1.0, abs(ref_losses[-1]))
      check_loss_decreased(result['losses'],
                           f"shard pipe={pipe} mp={model}, owned params {result['owned']}")
  print("    every shard matches its single-process reference  OK")


if __name__ == '__main__':
  tests = [
      ('matrix functions', test_compute_power),
      ('single process (CPU)', test_single_process_cpu),
      ('merged shapes', test_merged_shapes),
      ('embedding skip', test_embedding_skip),
      ('single process (GPU)', test_single_process_gpu),
      ('data parallel (gloo)', test_data_parallel),
      ('3D parallelism, 8 emulated ranks (gloo)', test_3d_topology),
  ]
  failed = []
  for name, fn in tests:
    print(f"[{name}]")
    try:
      fn()
    except Exception as e:
      import traceback; traceback.print_exc()
      failed.append(name)
  print()
  if failed:
    print(f"FAILED: {failed}")
    sys.exit(1)
  print(f"All {len(tests)} test groups passed.")
