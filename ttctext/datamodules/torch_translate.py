import re
import unicodedata
from io import BytesIO
from typing import *
from urllib.request import urlopen
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab, build_vocab_from_iterator

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
MAX_LENGTH = 10

special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

eng_prefixes = (
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readPairs(langzip: ZipFile, lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = (
        langzip.open(f"data/{lang1}-{lang2}.txt")
        .read()
        .decode("utf-8")
        .strip()
        .split("\n")
    )

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    return pairs


def filterPair(p):
    return (
        len(p[0].split(" ")) < MAX_LENGTH
        and len(p[1].split(" ")) < MAX_LENGTH
        and p[1].startswith(eng_prefixes)
    )


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def indexesFromSentence(lang: Vocab, sentence):
    return [lang.get_stoi()[word] for word in sentence.split(" ")]


def tensorFromSentence(lang: Vocab, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long)


def tensorsFromPair(lang1: Vocab, lang2: Vocab, pair):
    input_tensor = tensorFromSentence(lang1, pair[0])
    target_tensor = tensorFromSentence(lang2, pair[1])
    return (input_tensor, target_tensor)


class TorchLanguageData(pl.LightningDataModule):
    """
    DataModule for PyTorch Example Dataset, train, val, test splits and transforms
    Source: https://download.pytorch.org/tutorial/data.zip
    """

    name = "PyTorch example language dataset"
    zip_url = "https://download.pytorch.org/tutorial/data.zip"

    def __init__(
        self,
        lang_file: str = "eng-fra",
        reverse=True,
        val_split: float = 0.30,
        num_workers: int = 2,
        batch_size: int = 64,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how much the training pairs to use for the validation split, between (0 to 1)
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: desired batch size.
        """
        super().__init__(*args, **kwargs)

        assert (
            lang_file == "eng-fra"
        ), "eng-fra is the only choice for now, use `reverse=True` to get fra, eng pairs"

        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.input_lang, self.output_lang, self.l_pairs = self.prepare_langs(
            lang_file, reverse
        )

    def prepare_langs(self, lang_file="eng-fra", reverse=True):
        with urlopen(self.zip_url) as f:
            with BytesIO(f.read()) as b, ZipFile(b) as datazip:
                lang1, lang2 = lang_file.split("-")
                pairs = readPairs(datazip, lang1, lang2, reverse)

        print("Read %s sentence pairs" % len(pairs))
        pairs = filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        input_sentences, target_sentences = zip(*pairs)

        input_lang = build_vocab_from_iterator(
            [sentence.split(" ") for sentence in input_sentences],
            specials=special_tokens,
        )

        output_lang = build_vocab_from_iterator(
            [sentence.split(" ") for sentence in target_sentences],
            specials=special_tokens,
        )

        setattr(input_lang, "name", lang2 if reverse else lang1)
        setattr(output_lang, "name", lang1 if reverse else lang2)

        setattr(input_lang, "n_words", len(input_lang))
        setattr(output_lang, "n_words", len(output_lang))

        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        return input_lang, output_lang, pairs

    def prepare_data(self):
        """NOOP"""
        pass

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset"""
        print('splitting dataset into train and test"')
        self.train_pairs, self.val_pairs = train_test_split(
            self.l_pairs, test_size=self.val_split, random_state=69
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_pairs,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_pairs,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator_fn,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.val_pairs,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collator_fn,
        )
        return loader

    @property
    def collator_fn(self):
        def wrapper(batch):
            raw_src, raw_tgt = zip(*batch)
            src_batch, tgt_batch = zip(
                *[tensorsFromPair(self.input_lang, self.output_lang, x) for x in batch]
            )

            src_batch = torch.nn.utils.rnn.pad_sequence(
                src_batch, padding_value=PAD_token, batch_first=True
            )
            tgt_batch = torch.nn.utils.rnn.pad_sequence(
                tgt_batch, padding_value=PAD_token, batch_first=True
            )

            return src_batch, tgt_batch, raw_src, raw_tgt

        return wrapper
