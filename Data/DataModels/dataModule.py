import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, val_sequences, test_sequences, dataset: Dataset, batch_size: int = 8):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        self.dataset = dataset

    def setup(self, stage = None):
        self.train_dataset = self.dataset(self.train_sequences)
        self.val_dataset = self.dataset(self.val_sequences)
        self.test_dataset = self.dataset(self.test_sequences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2,
            persistent_workers=True
        )