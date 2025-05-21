import torch
from torch.utils.data import Dataset
from src.vocab import Vocab

class TransliterationDataset(Dataset):
    def __init__(self, path, input_vocab=None, target_vocab=None):
        self.pairs = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                hi, en = line.strip().split('\t')
                self.pairs.append((en, hi))

        self.input_vocab = input_vocab or Vocab()
        self.target_vocab = target_vocab or Vocab()

        if input_vocab is None or target_vocab is None:
            self.input_vocab.build_vocab([en for en, _ in self.pairs])
            self.target_vocab.build_vocab([hi for _, hi in self.pairs])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, hi = self.pairs[idx]
        return (
            torch.tensor(self.input_vocab.encode(en), dtype=torch.long),
            torch.tensor(self.target_vocab.encode(hi), dtype=torch.long),
        )
