PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

class Vocab:
    def __init__(self):
        self.char2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2char = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.num_chars = 4

    def build_vocab(self, words):
        for word in words:
            for char in word:
                if char not in self.char2idx:
                    self.char2idx[char] = self.num_chars
                    self.idx2char[self.num_chars] = char
                    self.num_chars += 1

    def encode(self, word):
        return [self.char2idx[SOS_TOKEN]] + [
            self.char2idx.get(c, self.char2idx[UNK_TOKEN]) for c in word
        ] + [self.char2idx[EOS_TOKEN]]

    def decode(self, indices):
        return ''.join([
            self.idx2char[i] for i in indices if i not in (self.char2idx[PAD_TOKEN], self.char2idx[SOS_TOKEN], self.char2idx[EOS_TOKEN])
        ])
