from torch.utils.data import Dataset


class CustomConcatDataset(Dataset):
    """
    A custom PyTorch Dataset class that concatenates multiple datasets.
    📚 Useful for combining data from different sources while maintaining
    the Dataset interface.
    """

    def __init__(self, datasets):
        """
        Constructor for the CustomConcatDataset class.

        :param datasets: A list of PyTorch Dataset objects to concatenate.
        """
        self.datasets = datasets

    def __len__(self):
        """
        Calculate the total length of the concatenated datasets.
        🔢 Returns the sum of the lengths of all individual datasets.

        :return: The total length of the concatenated datasets.
        """
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        """
        Get an item from the concatenated datasets based on the provided index.
        🔍 It finds the correct dataset and the corresponding index in that
        dataset.

        :param idx: The index of the desired item in the concatenated datasets.
        :return: The item corresponding to the given index.
        """
        dataset_idx = idx % len(self.datasets)
        item_idx = idx // len(self.datasets)
        return self.datasets[dataset_idx][item_idx]
