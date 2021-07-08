from typing import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import ttctext.datasets.utils.functional as text_f
from ttctext.augmentation import random_deletion, random_swap
from ttctext.datasets.sst import StanfordSentimentTreeBank


class SSTDataModule(pl.LightningDataModule):
    """
    DataModule for SST, train, val, test splits and transforms
    """

    name = "stanford_sentiment_treebank"

    def __init__(
        self,
        data_dir: str = ".",
        val_split: int = 1000,
        num_workers: int = 2,
        batch_size: int = 64,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: desired batch size.
        """
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...

        self.SST = StanfordSentimentTreeBank

    def prepare_data(self):
        """Saves IMDB files to `data_dir`"""
        self.SST(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset"""

        train_trans, test_trans = self.default_transforms

        train_dataset = self.SST(self.data_dir, split="train", **train_trans)
        test_dataset = self.SST(self.data_dir, split="test", **test_trans)

        train_length = len(train_dataset)

        self.raw_dataset_train = train_dataset
        self.raw_dataset_test = test_dataset

        # self.dataset_train, self.dataset_val = random_split(train_dataset, [train_length - self.val_split, self.val_split])
        self.dataset_train = train_dataset
        self.dataset_test = test_dataset

    def train_dataloader(self):
        """IMDB train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator_fn,
        )
        return loader

    def val_dataloader(self):
        """IMDB val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator_fn,
        )
        return loader

    def test_dataloader(self):
        """IMDB test set uses the test split"""
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator_fn,
        )
        return loader

    def get_vocab(self):
        return self.raw_dataset_train.get_vocab()

    @property
    def default_transforms(self):
        train_transforms = {
            "text_transforms": text_f.sequential_transforms(
                random_deletion, random_swap
            ),
            "label_transforms": None,
        }
        test_transforms = {"text_transforms": None, "label_transforms": None}

        return train_transforms, test_transforms

    @property
    def collator_fn(self):
        return self.raw_dataset_train.collator_fn
