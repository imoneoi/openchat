import pyarrow.parquet as pq
import orjson


class ParquetDataset:
    def __init__(self, dataset_filename):
        super().__init__()

        self.dataset = pq.read_table(dataset_filename)

        # read metadata
        self.metadata = self.dataset.schema.metadata.get(b"metadata_json", None)
        if self.metadata is not None:
            self.metadata = orjson.loads(self.metadata)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, str):
            return self.dataset.column(indices).to_numpy()
        
        return self.dataset.take(indices)
