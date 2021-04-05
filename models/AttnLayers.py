from torch import nn


class Attention(nn.Module) :
    def __init__(self, hidden_size) :
        super(Attention, self).__init__()
        self.attn1 = nn.Linear(hidden_size*2, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()


    def masked_softmax(self, attn_odds, masks) :
        attn_odds.masked_fill_(masks, -float('inf'))
        attn = nn.Softmax(dim=-1)(attn_odds)
        return attn
        
    def forward(self, hidden, masks) :
        attn1 = self.tanh(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = self.masked_softmax(attn2, masks)
        
        return attn
