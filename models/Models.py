import torch.nn as nn
from models.Encoders import EncoderRNN
from models.Decoders import AttnDecoder, FrozenAttnDecoder, PretrainedWeightsDecoder

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, emb, encoder, attention, use_attention):
        super(Model, self).__init__()
        self.encoder = EncoderRNN(vocab_size, embed_size, hidden_size, emb)
        if attention == 'frozen':
            self.decoder = FrozenAttnDecoder(hidden_size, output_size, use_attention, attention)
        elif attention == 'tanh':
            self.decoder = AttnDecoder(hidden_size, output_size, use_attention, attention)


    def forward(self, x, lens, masks):
        out, last = self.encoder(x, lens)
        out, attns = self.decoder(x, out, last, lens, masks)
        return out, attns
