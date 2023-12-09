import numpy as np
import numba


@numba.njit
def ffd_check(a: np.ndarray, c: int, n: int):
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins = np.full((n, ), c, dtype=a.dtype)
    for size in a:
        not_found = True
        for idx in range(n):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        if not_found:
            return False

    return True


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
def allocate(lengths: np.ndarray, numseqs: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    s = 0
    start_index = 0
    result = []
    result_totseqs = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while r - l > 1:
            m = (l + r) // 2
            if ffd_check(lengths[start_index: start_index + m], c, n):
                l = m
            else:
                r = m

        # use length l
        batch = ffd_with_result(lengths[start_index: start_index + l], c, start_index)  # type: ignore
        if len(batch) < n:
            break

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])
        # add total seqs for all ranks
        totseq = 0
        for indices in batch:
            for idx in indices:
                totseq += numseqs[idx]
        result_totseqs.append(totseq)

    return result, result_totseqs, s, len(result) * c * n


class MultipackDistributedSampler:
    """Unpadded data loading using Multipack.
       Approximate (at most ~1.22x) the optimal solution of the identical-machines scheduling problem, which is NP-hard."""
    
    def __init__(
        self,
        lengths: np.ndarray,
        numseqs: np.ndarray,

        batch_max_length: int,

        num_replicas: int,
        rank: int,

        seed: int,
    ):
        # Dataset
        self.lengths = lengths
        self.numseqs = numseqs
        assert isinstance(self.lengths, np.ndarray)

        self.batch_max_length = batch_max_length

        # Get rank
        self.num_replicas = num_replicas
        self.rank = rank

        # Seed
        self.seed = seed

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

    def generate_batches(self, epoch, set_stats=False):
        indices = np.random.default_rng(seed=self.seed + epoch).permutation(len(self.lengths))

        lengths        = self.lengths[indices]
        numseqs        = self.numseqs[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, totseqs, total_used, total_slots = allocate(lengths=lengths,
                                                             numseqs=numseqs,
                                                             lengths_cumsum=lengths_cumsum,
                                                             rank=self.rank,
                                                             c=self.batch_max_length,
                                                             n=self.num_replicas)
        
        curseqs = [np.sum(numseqs[batch]) for batch in batches]
        batches = [indices[batch]         for batch in batches]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots

        return batches, totseqs, curseqs
    
    def iter(self, epoch):
        all_batches, all_totseqs, all_curseqs = self.generate_batches(epoch, set_stats=True)

        for batch, totseq, curseq in zip(all_batches, all_totseqs, all_curseqs):
            yield batch, totseq, curseq

    def estimate_num_batches(self):
        batches, _, _ = self.generate_batches(epoch=0)
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots
