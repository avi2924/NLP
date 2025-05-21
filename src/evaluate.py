import torch
import pickle
from src.model import Encoder, Decoder
from src.dataset import TransliterationDataset
from src.config import CONFIG
from utils.utils import collate_fn
from torch.utils.data import DataLoader

def load_model(input_vocab_size, target_vocab_size):
    encoder = Encoder(input_vocab_size, CONFIG["embedding_size"], CONFIG["hidden_size"]).to(CONFIG["device"])
    decoder = Decoder(target_vocab_size, CONFIG["embedding_size"], CONFIG["hidden_size"]).to(CONFIG["device"])
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"])
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def predict(encoder, decoder, word, input_vocab, target_vocab, max_len=20):
    with torch.no_grad():
        input_seq = torch.tensor([input_vocab.encode(word)], dtype=torch.long).to(CONFIG["device"])
        enc_outputs, h, c = encoder(input_seq)
        dec_input = torch.tensor([target_vocab.char2idx["<SOS>"]], dtype=torch.long).to(CONFIG["device"])
        dec_hidden = (h[:1], c[:1])
        predicted_indices = []

        for _ in range(max_len):
            output, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_outputs)
            top1 = output.argmax(1)
            if top1.item() == target_vocab.char2idx["<EOS>"]:
                break
            predicted_indices.append(top1.item())
            dec_input = top1

        return target_vocab.decode(predicted_indices)

def evaluate():
    # Load vocabs from disk
    with open(CONFIG["input_vocab_path"], "rb") as f:
        input_vocab = pickle.load(f)
    with open(CONFIG["target_vocab_path"], "rb") as f:
        target_vocab = pickle.load(f)
    # Use these vocabs to create the dataset and models
    dataset = TransliterationDataset(CONFIG["test_path"], input_vocab=input_vocab, target_vocab=target_vocab)
    encoder, decoder = load_model(len(input_vocab.char2idx), len(target_vocab.char2idx))

    correct = 0
    total = len(dataset.pairs)

    for i in range(total):
        en, hi = dataset.pairs[i]
        pred = predict(encoder, decoder, en, dataset.input_vocab, dataset.target_vocab)
        print(f"EN: {en} | GT: {hi} | PRED: {pred}")
        if pred == hi:
            correct += 1

    accuracy = correct / total * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate()