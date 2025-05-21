import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import collate_fn
from src.model import Encoder, Decoder
from src.dataset import TransliterationDataset
from src.config import CONFIG
import pickle

def calculate_accuracy(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            enc_outputs, h, c = encoder(src)
            dec_input = tgt[:, 0]
            dec_hidden = (h[:1], c[:1])
            batch_size = tgt.size(0)
            max_len = tgt.size(1)
            outputs = []
            for t in range(1, max_len):
                dec_output, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_outputs)
                top1 = dec_output.argmax(1)
                outputs.append(top1.unsqueeze(1))
                dec_input = top1
            outputs = torch.cat(outputs, dim=1)  # [batch, seq_len-1]
            # Compare outputs with tgt[:, 1:]
            for i in range(batch_size):
                pred = outputs[i].tolist()
                true = tgt[i, 1:].tolist()
                # Stop at first EOS (0) if using padding/EOS
                if 0 in true:
                    true = true[:true.index(0)]
                if 0 in pred:
                    pred = pred[:pred.index(0)]
                if pred == true:
                    correct += 1
                total += 1
    encoder.train()
    decoder.train()
    return correct / total if total > 0 else 0

def train():
    dataset = TransliterationDataset(CONFIG["train_path"])
    # Save vocabs for later use in evaluation
    with open(CONFIG["input_vocab_path"], "wb") as f:
        pickle.dump(dataset.input_vocab, f)
    with open(CONFIG["target_vocab_path"], "wb") as f:
        pickle.dump(dataset.target_vocab, f)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], collate_fn=collate_fn, shuffle=True)

    # Validation set
    val_dataset = TransliterationDataset(CONFIG["dev_path"], input_vocab=dataset.input_vocab, target_vocab=dataset.target_vocab)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], collate_fn=collate_fn)

    encoder = Encoder(len(dataset.input_vocab.char2idx), CONFIG["embedding_size"], CONFIG["hidden_size"]).to(CONFIG["device"])
    decoder = Decoder(len(dataset.target_vocab.char2idx), CONFIG["embedding_size"], CONFIG["hidden_size"]).to(CONFIG["device"])

    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=CONFIG["learning_rate"])
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=CONFIG["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0
        for src, tgt in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(CONFIG["device"]), tgt.to(CONFIG["device"])
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            enc_outputs, h, c = encoder(src)
            dec_input = tgt[:, 0]
            dec_hidden = (h[:1], c[:1])
            loss = 0

            for t in range(1, tgt.size(1)):
                dec_output, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_outputs)
                loss += criterion(dec_output, tgt[:, t])
                dec_input = tgt[:, t]

            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            total_loss += loss.item() / tgt.size(1)

        train_acc = calculate_accuracy(encoder, decoder, dataloader, CONFIG["device"])
        val_acc = calculate_accuracy(encoder, decoder, val_dataloader, CONFIG["device"])
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f} | Train Acc = {train_acc*100:.2f}% | Val Acc = {val_acc*100:.2f}%")
        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, CONFIG["checkpoint_path"])

if __name__ == "__main__":
    train()