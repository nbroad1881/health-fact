#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" 
Model class for combing transformers and LSTMs for long sequences 

Work-in-progress
"""


from dataclasses import dataclass

import torch
from torch import nn

from transformers import PreTrainedModel, AutoModel
from transformers.modeling_utils import ModelOutput


class BERTLSTM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            config.hidden_size,
            (config.hidden_size) // 2,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, doc_ids=None, labels=None):
        """
        This expects a few different documents. The doc_ids indicate which document each sequence
        comes from. 
        There is one label for each document.
        """

        sequence_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]

        sequence_output = self.dropout(sequence_output)

        id_mode, _ = torch.mode(doc_ids)
        max_length = (doc_ids == id_mode).sum()

        num_items = doc_ids.unique()

        # [number of docs, max_number_of_chunks, hidden_size]
        lstm_input = torch.zeros(
            size=(num_items.numel(), max_length, self.config.hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )

        # doc_ids can be any value
        # by subtracting the max, the doc_ids now are indexes
        # into sequence_output
        doc_ids = doc_ids - doc_ids.min()
        true_labels = torch.zeros(
            size=(doc_ids.max().item()+1, 1), 
            dtype=torch.long,
            device=input_ids.device
        )
        for id_ in range(0, doc_ids.max().item() + 1):

            mask = doc_ids == id_
            seq_len = (mask).sum()  # how many chunks the doc was broken into

            # stick the CLS embeddings into a sequence for each doc
            lstm_input[id_, :seq_len, :] = sequence_output[mask, 0, :]
            true_labels[id_] = labels[mask].unique()

        lstm_output, hc = self.bilstm(lstm_input)

        logits = self.classifier(lstm_output).mean(dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), true_labels.view(-1))

        return SequenceClassificationOutput(
            loss=loss,
            logits=logits,
        )


@dataclass
class SequenceClassificationOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class BERTLSTM2(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.transformer = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            config.hidden_size,
            (config.hidden_size) // 2,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, label=None):
        """
        This expects a single document and a single label.
        """

        sequence_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]

        sequence_output = self.dropout(sequence_output)

        lstm_output, hc = self.bilstm(sequence_output)

        logits = self.classifier(lstm_output)

        loss = None
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), label.view(-1))

        return SequenceClassificationOutput(
            loss=loss,
            logits=logits,
        )