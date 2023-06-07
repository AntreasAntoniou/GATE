import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from torch.utils.data import Dataset


# Assume you have a PyTorch Dataset
class MyDataset(Dataset):
    def __init__(self):
        self.data = [{"image": i, "label": i % 2} for i in range(1000)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Instantiate your PyTorch dataset
pytorch_dataset = MyDataset()


def convert_to_parquet(pytorch_dataset, parquet_file_path):
    """
    Convert a PyTorch dataset to a Parquet file.

    Args:
        pytorch_dataset (torch.utils.data.Dataset): The PyTorch dataset to convert.
        parquet_file_path (str): The path where the Parquet file will be saved.
    """
    # Convert the PyTorch dataset to a PyArrow Table
    data = [sample for sample in pytorch_dataset]
    table = pa.Table.from_pandas(pd.DataFrame(data))

    # Write the Table to a Parquet file
    pq.write_table(table, parquet_file_path)


# Convert the PyTorch dataset to a Parquet file
parquet_file_path = "my_dataset.parquet"
convert_to_parquet(pytorch_dataset, parquet_file_path)

# Load the Parquet file as a HuggingFace dataset
hf_dataset = load_dataset("parquet", data_files=parquet_file_path)

print(hf_dataset)
