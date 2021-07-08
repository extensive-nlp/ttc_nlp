from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import gdown
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from tqdm.auto import tqdm

import ttctext.datasets.utils.functional as text_f


def build_vocab_from_iterator(iterator, num_lines=None):
    """
    Build a Vocab from an iterator.
    Args:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        num_lines: The expected number of elements returned by the iterator.
            (Default: None)
            Optionally, if known, the expected number of elements can be passed to
            this factory function for improved progress reporting.
    """

    counter = Counter()
    with tqdm(unit_scale=0, unit="lines", total=num_lines) as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    word_vocab = Vocab(counter)
    return word_vocab


class StanfordSentimentTreeBank(Dataset):
    """The Standford Sentiment Tree Bank Dataset
    Stanford Sentiment Treebank V1.0

    This is the dataset of the paper:

    Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
    Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
    Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)

    If you use this dataset in your research, please cite the above paper.

    @incollection{SocherEtAl2013:RNTN,
    title = {{Parsing With Compositional Vector Grammars}},
    author = {Richard Socher and Alex Perelygin and Jean Wu and Jason Chuang and Christopher Manning and Andrew Ng and Christopher Potts},
    booktitle = {{EMNLP}},
    year = {2013}
    }
    """

    ORIG_URL = "http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
    DATASET_NAME = "StanfordSentimentTreeBank"
    URL = "https://drive.google.com/uc?id=1urNi0Rtp9XkvkxxeKytjl1WoYNYUEoPI"
    OUTPUT = "sst_dataset.zip"

    def __init__(
        self,
        root,
        vocab=None,
        text_transforms=None,
        label_transforms=None,
        split="train",
        ngrams=1,
        use_transformed_dataset=True,
    ) -> None:
        """Initiate text-classification dataset.
        Args:
            data: a list of label and text tring tuple. label is an integer.
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of label and text string transforms.
        """

        super(self.__class__, self).__init__()

        if split not in ["train", "test"]:
            raise ValueError(
                f'split must be either ["train", "test"] unknown split {split}'
            )

        self.vocab = vocab

        gdown.cached_download(self.URL, Path(root) / self.OUTPUT)

        self.generate_sst_dataset(split, Path(root) / self.OUTPUT)

        tokenizer = get_tokenizer("basic_english")

        # the text transform can only work at the sentence level
        # the rest of tokenization and vocab is done by this class
        self.text_transform = text_f.sequential_transforms(
            tokenizer, text_f.ngrams_func(ngrams)
        )

        def build_vocab(data, transforms):
            def apply_transforms(data):
                for line in data:
                    yield transforms(line)

            return build_vocab_from_iterator(apply_transforms(data), len(data))

        if self.vocab is None:
            # vocab is always built on the train dataset
            self.vocab = build_vocab(self.dataset_train["phrase"], self.text_transform)

        if text_transforms is not None:
            self.text_transform = text_f.sequential_transforms(
                self.text_transform,
                text_transforms,
                text_f.vocab_func(self.vocab),
                text_f.totensor(dtype=torch.long),
            )
        else:
            self.text_transform = text_f.sequential_transforms(
                self.text_transform,
                text_f.vocab_func(self.vocab),
                text_f.totensor(dtype=torch.long),
            )

        self.label_transform = text_f.sequential_transforms(
            text_f.totensor(dtype=torch.long)
        )

    def generate_sst_dataset(self, split: str, dataset_file: Path) -> None:

        with ZipFile(dataset_file) as datasetzip:
            with datasetzip.open("sst_dataset/sst_dataset_augmented.csv") as f:
                dataset = pd.read_csv(f, index_col=0)

        self.dataset_orig = dataset.copy()

        dataset_train_raw = dataset[dataset["splitset_label"].isin([1, 3])]
        self.dataset_train = pd.concat(
            [
                dataset_train_raw[["phrase_cleaned", "sentiment_values"]].rename(
                    columns={"phrase_cleaned": "phrase"}
                ),
                dataset_train_raw[["synonym_sentences", "sentiment_values"]].rename(
                    columns={"synonym_sentences": "phrase"}
                ),
                dataset_train_raw[["backtranslated", "sentiment_values"]].rename(
                    columns={"backtranslated": "phrase"}
                ),
            ],
            ignore_index=True,
        )

        if split == "train":
            self.dataset = self.dataset_train.copy()
        else:
            self.dataset = (
                dataset[dataset["splitset_label"].isin([2])][
                    ["phrase_cleaned", "sentiment_values"]
                ]
                .rename(columns={"phrase_cleaned": "phrase"})
                .reset_index(drop=True)
            )

    @staticmethod
    def discretize_label(label: float) -> int:
        if label <= 0.2:
            return 0
        if label <= 0.4:
            return 1
        if label <= 0.6:
            return 2
        if label <= 0.8:
            return 3
        return 4

    def __getitem__(self, idx):
        # print(f'text: {self.dataset["sentence"].iloc[idx]}, label: {self.dataset["sentiment_values"].iloc[idx]}')
        text = self.text_transform(self.dataset["phrase"].iloc[idx])
        label = self.label_transform(self.dataset["sentiment_values"].iloc[idx])
        # print(f't_text: {text} {text.shape}, t_label: {label}')
        return label, text

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_labels():
        return ["very negative", "negative", "neutral", "positive", "very positive"]

    def get_vocab(self):
        return self.vocab

    @property
    def collator_fn(self):
        def collate_fn(batch):
            pad_idx = self.get_vocab()["<pad>"]

            labels, sequences = zip(*batch)

            labels = torch.stack(labels)

            lengths = torch.LongTensor([len(sequence) for sequence in sequences])

            # print('before padding: ', sequences[40])

            sequences = torch.nn.utils.rnn.pad_sequence(
                sequences, padding_value=pad_idx, batch_first=True
            )
            # print('after padding: ', sequences[40])

            return labels, sequences, lengths

        return collate_fn
