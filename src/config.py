import os
import torch
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CONFIG = {
    "train_path": os.path.join(BASE_DIR, "data/dataset/hi/lexicons/train_new.tsv"),
    "dev_path": os.path.join(BASE_DIR, "data/dataset/hi/lexicons/dev_new.tsv"),
    "test_path": os.path.join(BASE_DIR, "data/dataset/hi/lexicons/test_new.tsv"),
    "batch_size": 64,
    "hidden_size": 256,
    "embedding_size": 128,
    "num_layers": 2,
    "learning_rate": 0.001,
    "num_epochs": 20,
    "checkpoint_path": os.path.join(BASE_DIR, "checkpoints/model.pt"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_vocab_path": "input_vocab.pkl",
    "target_vocab_path": "target_vocab.pkl"
}
