from dataclasses import dataclass

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


@dataclass
class DataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        self.raw_dataset = load_dataset("health_fact")
        self.raw_dataset = self.raw_dataset.filter(lambda x: x["label"] != -1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_name_or_path"],
            use_auth_token=True,
        )

    def prepare_dataset(self):

        def tokenize(example):
            return self.tokenizer(
                example["claim"],
                example["main_text"],
                padding=False,
                truncation="only_second",
                max_length=self.cfg["max_seq_length"],
            )

        cols = self.raw_dataset["train"].column_names
        self.tokenized_dataset = self.raw_dataset.map(
            tokenize, batched=False, num_proc=self.cfg["num_proc"], remove_columns=cols
        )