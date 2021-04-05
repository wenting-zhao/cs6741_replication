import torch
import torch.nn as nn

class EncoderRNN(nn.Module) :
    def __init__(self, vocab_size, embed_size, hidden_size, emb=None):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if emb is not None:
            weight = torch.Tensor(emb)
            weight[0, :].zero_()
            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)  
            print("loaded pre-emb")
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.output_size = self.hidden_size * 2

    def forward(self, data, lengths) :
        embedding = self.embedding(data) #(B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)
        last_hidden = torch.cat([h[0], h[1]], dim=-1)
        return output, last_hidden
