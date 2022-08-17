from typing import Dict, List
from dataclasses import dataclass

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class DataModule:

    cfg: Dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.model_name_or_path,
        )
        self.label2id = {l:i for i, l in enumerate(self.cfg.data.labels)}
        self.id2label = {i:l for l, i in self.label2id.items()}

    def prepare_dataset(self) -> None:
        """
        Load in dataset and tokenize.

        If debugging, take small subset of the full dataset.
        """
        self.raw_dataset = load_dataset(self.cfg.data.dataset_name, self.cfg.data.dataset_config_name)
        self.raw_dataset = self.raw_dataset.filter(lambda x: x["label"] != -1)

        if self.cfg.data.n_rows > 0:
            for split in self.raw_dataset:
                self.raw_dataset[split] = self.raw_dataset[split].select(
                    range(self.cfg.data.n_rows)
                )

        cols = self.raw_dataset["train"].column_names
        self.tokenized_dataset = self.raw_dataset.map(
            self.tokenize,
            batched=True,
            num_proc=self.cfg.num_proc,
            remove_columns=cols,
            with_indices=True,
        )

    def tokenize(self, examples: Dict[str, str], indices: List[str]):
        """
        Tokenize texts by putting claim text in front of main text.

        If using a small model, can stride over the text.
        """

        tokenized = self.tokenizer(
            examples["claim"],
            examples["main_text"],
            padding=False,
            truncation=self.cfg.data.truncation,
            max_length=self.cfg.data.max_seq_length,
        )

        tokenized["labels"] = examples["label"]

        return tokenized

    def get_train_dataset(self, tokenized: bool = True) -> datasets.Dataset:
        if tokenized:
            return self.tokenized_dataset["train"]
        return self.raw_dataset["train"]

    def get_eval_dataset(self, tokenized: bool = True) -> datasets.Dataset:
        if tokenized:
            return self.tokenized_dataset["validation"]
        return self.raw_dataset["validation"]

    def get_test_dataset(self, tokenized: bool = True) -> datasets.Dataset:
        if tokenized:
            return self.tokenized_dataset["test"]
        return self.raw_dataset["test"]
