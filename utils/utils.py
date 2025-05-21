from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    return src_padded, tgt_padded
