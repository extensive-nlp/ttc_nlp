import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy


class SSTModel(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.num_classes = self.hparams.output_dim

        self.embedding = nn.Embedding(
            self.hparams.input_dim, self.hparams.embedding_dim
        )

        self.lstm = nn.LSTM(
            self.hparams.embedding_dim,
            self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )

        self.proj_layer = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.BatchNorm1d(self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
        )

        self.fc = nn.Linear(self.hparams.hidden_dim, self.num_classes)

        self.loss = nn.CrossEntropyLoss()

    def init_state(self, sequence_length):
        return (
            torch.zeros(
                self.hparams.num_layers, sequence_length, self.hparams.hidden_dim
            ).to(self.device),
            torch.zeros(
                self.hparams.num_layers, sequence_length, self.hparams.hidden_dim
            ).to(self.device),
        )

    def forward(self, text, text_length, prev_state=None):
        # [batch size, sentence length] => [batch size, sentence len, embedding size]
        embedded = self.embedding(text)

        # packs the input for faster forward pass in RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_length.to("cpu"), enforce_sorted=False, batch_first=True
        )

        # [batch size sentence len, embedding size] =>
        #   output: [batch size, sentence len, hidden size]
        #   hidden: [batch size, 1, hidden size]
        packed_output, curr_state = self.lstm(packed, prev_state)

        hidden_state, cell_state = curr_state

        # print('hidden state shape: ', hidden_state.shape)
        # print('cell')

        # unpack packed sequence
        # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # print('unpacked: ', unpacked.shape)

        # [batch size, sentence len, hidden size] => [batch size, num classes]
        # output = self.proj_layer(unpacked[:, -1])
        output = self.proj_layer(hidden_state[-1])

        # print('output shape: ', output.shape)

        output = self.fc(output)

        return output, curr_state

    def shared_step(self, batch, batch_idx):
        label, text, text_length = batch

        logits, in_state = self(text, text_length)

        loss = self.loss(logits, label)

        pred = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(pred, label)

        metric = {"loss": loss, "acc": acc}

        return metric

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)

        log_metrics = {"train_loss": metrics["loss"], "train_acc": metrics["acc"]}

        self.log_dict(log_metrics, prog_bar=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)

        return metrics

    def validation_epoch_end(self, outputs):
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        log_metrics = {"val_loss": loss, "val_acc": acc}

        self.log_dict(log_metrics, prog_bar=True)

        return log_metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        accuracy = torch.stack([x["acc"] for x in outputs]).mean()

        self.log("hp_metric", accuracy)

        self.log_dict({"test_acc": accuracy}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, verbose=True
            ),
            "monitor": "train_loss",
            "name": "scheduler",
        }
        return [optimizer], [lr_scheduler]
