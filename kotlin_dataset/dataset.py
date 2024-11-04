import json

from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset


class KotlinDataset(Dataset):
    def __init__(self, folder_path: str, tokenizer: PreTrainedTokenizer):
        self.inputs = []
        self.tokenizer = tokenizer
        with open(folder_path) as file:
            for i, line in enumerate(file):
                self.inputs.append(self.tokenize_fn(json.loads(line)))

    def __len__(self):
        return len(self.inputs)

    def tokenize_fn(self, line):
        input_problem = line["problem"]
        input_solution = line["solution"]

        inputs = self.tokenizer(input_problem)
        targets = self.tokenizer(input_solution)
        inputs["labels"] = [-100] * len(inputs["input_ids"]) + targets[
            "input_ids"
        ]  # we don't train on the problem tokens
        inputs["input_ids"] = inputs["input_ids"] + targets["input_ids"]
        inputs["attention_mask"] = inputs["attention_mask"] + targets["attention_mask"]

        return inputs

    def __getitem__(self, idx):
        return self.inputs[idx]
