import pyarrow.parquet as pq
import orjson


class NumpyDataset:
    def __init__(self, dataset_filename):
        super().__init__()

        # Convert parquet to numpy for fast random access
        table = pq.read_table(dataset_filename, memory_map=True)
        self.dataset = {k: v.to_numpy() for k, v in zip(table.column_names, table.columns)}
        self.length = table.num_rows

        # read metadata
        self.metadata = table.schema.metadata.get(b"metadata_json", None)
        if self.metadata is not None:
            self.metadata = orjson.loads(self.metadata)

    def __len__(self):
        return self.length

    def __getitem__(self, indices):
        if isinstance(indices, str):
            return self.dataset[indices]
        
        return {k: v[indices] for k, v in self.dataset.items()}
