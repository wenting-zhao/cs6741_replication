import torch
import torch.nn as nn
from models.AttnLayers import Attention


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size=1, use_attention=True, attention='tanh'):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_1 = nn.Linear(hidden_size*2, output_size)

        self.attention = Attention(hidden_size)

        self.use_attention = use_attention
               
    def decode(self, predict) :
        predict = self.linear_1(predict)
        return predict

    def forward(self, data, z, last, lengths=None, masks=None):
        if self.use_attention:
            attn = self.attention(z, masks)

            context = (attn.unsqueeze(-1) * z).sum(1)
        else:
            attn = None
            context = last
            
        predict = self.decode(context)
        return predict, attn


class FrozenAttnDecoder(AttnDecoder):

    def generate_frozen_uniform_attn(self, data, lengths, masks):
        attn = torch.zeros((len(data), torch.max(lengths)))
        inv_l = 1. / (lengths - 2)
        attn += inv_l[:, None]
        attn = attn.cuda()
        attn.masked_fill_(masks, 0) 
        return attn

    def forward(self, data, z, last, lengths=None, masks=None):
        if self.use_attention:
            attn = self.generate_frozen_uniform_attn(data, lengths, masks)
            context = (attn.unsqueeze(-1) * z).sum(1)
        else:
            attn = None
            context = last
            
        predict = self.decode(context)
        return predict, attn


class PretrainedWeightsDecoder(AttnDecoder) :

    def forward(self, data) :
        if self.use_attention :
            output = data.hidden
            attn = data.target_attn

            context = (attn.unsqueeze(-1) * output).sum(1)
            data.attn = attn
        else :
            context = data.last_hidden
            
        predict = self.decode(context)
        data.predict = predict
