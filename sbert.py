from itertools import chain

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel


class SBertTransformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.sbert = AutoModel.from_pretrained(config.sbert_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            dim_feedforward=3072,
            nhead=12,
            batch_first=True,
            activation="gelu",
            layer_norm_eps=1e-7,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, lengths=None, ids=None
    ):
        """
        input_ids will be of shape [num_chunks, sequence_length] where num_chunks can be across multiple documents
        attention_mask is same shape as input_ids
        labels is [num_documents, num_classes] if multilabel, [num_documents] for multiclass or regression
        """

        # sbert is frozen, so no gradients needed
        with torch.no_grad():
            sbert_output = self.sbert(
                input_ids=input_ids, attention_mask=attention_mask
            )
            embeddings = self.mean_pooling(sbert_output, attention_mask)

        # embeddings will be of shape [num_chunks, hidden_dim]
        # Now need to be reshaped to separate documents

        max_seq_len = max(lengths)
        hidden_dim = embeddings.size(-1)

        docs = []
        start = 0
        for length in lengths:
            temp = embeddings[start : start + length, :]
            z = torch.zeros(
                (max_seq_len - length, hidden_dim), dtype=temp.dtype, device=temp.device
            )
            docs.append(torch.vstack([temp, z]).unsqueeze(0))
            start += length

        new_input = torch.vstack(docs)

        attention_mask = torch.all(new_input == 0, dim=-1).to(new_input.device)

        transformer_output = self.transformer(
            new_input, src_key_padding_mask=attention_mask
        )

        logits = self.classifier(transformer_output).mean(dim=1)

        loss = None
        if labels is not None:
            # TODO: Implement multilabel, regression losses

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class SBertCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        """
        This expects to get examples from the dataset.
        Each row in the dataset will have
            input_ids [num_chunks, sequence_length]
            attention_mask [num_chunks, sequence_length]
            label [num_classes]

        The text should already be tokenized and padded to the documents longest sequence.


        """

        label_key = "label" if "label" in features[0] else "labels"

        lengths = [len(x["input_ids"]) for x in features]
        ids = list(chain(*[x["input_ids"] for x in features]))
        mask = list(chain(*[x["attention_mask"] for x in features]))
        labels = [x[label_key] for x in features]

        longest_seq = max([len(x) for x in ids])

        ids = [x + [self.tokenizer.pad_token_id] * (longest_seq - len(x)) for x in ids]
        mask = [x + [0] * (longest_seq - len(x)) for x in mask]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "lengths": lengths,
        }
