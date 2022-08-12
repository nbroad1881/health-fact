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

        def tokenize(examples, indices):

            tokenized =  self.tokenizer(
                examples["claim"],
                examples["main_text"],
                padding=False,
                truncation=self.cfg["truncation"],
                stride=self.cfg["stride"],
                max_length=self.cfg["max_seq_length"],
                return_overflowing_tokens=self.cfg["stride"] is not None,
            )

            if self.cfg["stride"]:
                labels = []
                idxs = []

                for o in tokenized["overflow_to_sample_mapping"]:
                    idxs.append(indices[o])
                    labels.append(examples["label"][o])

                tokenized["idx"] = idxs
                tokenized["labels"] = labels
            else:
                tokenized["labels"] = examples["label"]
                

            return tokenized
        
        
        cols = self.raw_dataset["train"].column_names
        self.tokenized_dataset = self.raw_dataset.map(
            tokenize, 
            batched=True, 
            num_proc=self.cfg["num_proc"], 
            remove_columns=cols, 
            with_indices=True
        )