import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

import pyarrow.parquet as pq
import orjson

from ochat.training_deepspeed.multipack_sampler import MultipackDistributedSampler


def _find_multiple(a, b):
    return (-(a // -b)) * b


class OpenchatDataset(IterableDataset):
    def __init__(self, dataset_filename, batch_max_length, rank, num_replicas):
        super().__init__()
        # Init constants
        self.PAD_ID = 0
        self.PAD_MULTIPLE = 64
        self.BATCH_KEYS = {
            "seqlens": torch.int32,
            "nz_input_ids": torch.long,
            "nz_position_ids": torch.long,
            "nz_shifted_label_ids": torch.long,

            "nz_shifted_loss_weights": torch.bfloat16
        }

        assert batch_max_length % self.PAD_MULTIPLE == 0, f"Batch size {batch_max_length} need to be multiples of {self.PAD_MULTIPLE}"

        # Load data
        # Convert parquet to numpy for fast random access
        table = pq.read_table(dataset_filename, memory_map=True)
        self.dataset = {k: v.to_numpy() for k, v in zip(table.column_names, table.columns)}

        # read metadata
        self.metadata = table.schema.metadata.get(b"metadata_json", None)
        if self.metadata is not None:
            self.metadata = orjson.loads(self.metadata)

        # Free table space
        del table

        # Create sampler
        self.sampler = MultipackDistributedSampler(
            lengths=self.dataset["total_length"],
            numseqs=self.dataset["num_seqs"],

            batch_max_length=batch_max_length,

            rank=rank,
            num_replicas=num_replicas,
            seed=0
        )

        # Init state
        self._epoch = 0

    def _load_batch(self, indices):
        batch = {k: v[indices] for k, v in self.dataset.items()}

        # Concat batches
        batch = {k: np.concatenate(batch[k], axis=0) for k in self.BATCH_KEYS.keys()}

        # Pad an unused item to reach multiple of PAD_MULTIPLE, for faster GEMM
        total_seqlen = batch["nz_input_ids"].size
        pad_len      = _find_multiple(total_seqlen, self.PAD_MULTIPLE) - total_seqlen

        if pad_len > 0:
            assert pad_len < self.PAD_MULTIPLE

            # total length
            padding_specs = {
                "seqlens": (1, pad_len),

                "nz_input_ids": (pad_len, self.PAD_ID),
                "nz_position_ids": (pad_len, 0),
                "nz_shifted_label_ids": (pad_len, self.PAD_ID),
                "nz_shifted_loss_weights": (pad_len, 0),
            }
            for k, pad_spec in padding_specs.items():
                batch[k] = np.concatenate((batch[k], np.full(*pad_spec, dtype=batch[k].dtype)), axis=0)

        # to tensor
        batch_tensor = {}
        for k, dtype in self.BATCH_KEYS.items():
            batch_tensor[k] = torch.from_numpy(batch[k]).to(dtype)

        # cu seqlens
        batch_tensor["cu_seqlens"] = torch.nn.functional.pad(batch_tensor["seqlens"].cumsum(-1, dtype=torch.int32), (1, 0))
        # batch info
        batch_info = {"max_seqlen": torch.max(batch_tensor["seqlens"]).item()}

        # inputs
        del batch_tensor["seqlens"]
        return batch_tensor, batch_info

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1

        for indices, all_numseq, cur_numseq in self.sampler.iter(self._epoch):
            yield self._load_batch(indices), all_numseq, cur_numseq

        # Increase epoch count
        self._epoch += 1

    def estimate_num_batches(self):
        return self.sampler.estimate_num_batches()
