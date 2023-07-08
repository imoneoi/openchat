from typing import Optional, List

import torch.distributed as dist
from torch.utils.data import Sampler

import numpy as np
import numba


@numba.njit
def ffd(a: np.ndarray, c: int):
    # First-fit-decreasing bin packing
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins = []
    for size in a:
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                add_new = False
                break

        if add_new:
            bins.append(c - size)

    return len(bins)


@numba.njit
def ffd_with_result(a: np.ndarray, c: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)

    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins = []
    bins_result = []
    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break

        if add_new:
            bins.append(c - size)
            bins_result.append([indices[a_id] + start_index])

    return bins_result


@numba.njit
def allocate(lengths: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~96.4% efficiency on OpenChat training set (2048 ctx len)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while r - l > 1:
            m = (l + r) // 2
            if ffd(lengths[start_index: start_index + m], c) <= n:
                l = m
            else:
                r = m

        # use length l
        batch = ffd_with_result(lengths[start_index: start_index + l], c, start_index)
        if len(batch) < n:
            break

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])

    return result, s / max(1, len(result) * c * n)  # Avoid division by zero


class FFDDistributedBatchSampler(Sampler):
    """Unpadded length sampling using FFD (First-fit-decreasing bin packing).
       Approximate (at most ~1.22x) the optimal solution of the identical-machines scheduling problem, which is NP-hard."""
    
    def __init__(
        self,
        batch_max_length: int,
        lengths: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        # Get rank
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.batch_max_length = batch_max_length
        self.lengths = lengths
        assert isinstance(self.lengths, np.ndarray)

        self.epoch = 0

        # statistics
        self.total_epochs = 0
        self.total_efficiency = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = np.random.default_rng(seed=self.seed + self.epoch).permutation(len(self.lengths))

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, efficiency = allocate(lengths=lengths,
                           lengths_cumsum=lengths_cumsum,
                           rank=self.rank,
                           c=self.batch_max_length,
                           n=self.num_replicas)
        
        batches = [indices[batch] for batch in batches]

        # statistics
        if set_stats:
            self.total_epochs += 1
            self.total_efficiency += efficiency
        
        return batches
    
    def __iter__(self):
        batches = self.generate_batches(set_stats=True)
        return iter(batches)

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)

    def efficiency(self):
        return self.total_efficiency / self.total_epochs
