import torch
import torch.nn as nn
import config

class Decoder(nn.Module):
    def __init__(self, vocab_dim, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = config.device
        self.embedding = nn.Embedding(self.vocab_dim, self.embed_dim)\
                            .to(self.device)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim).to(self.device)
        self.lin = nn.Linear(self.hidden_dim, self.vocab_dim).to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)
        
    def forward(self, word, state):
        hidden = state[0].view(1, 1, -1)
        enc_context = state[1].view(1, 1, -1)
        result = self.embedding(word)
        result = result.view(1, 1, self.embed_dim)
        result,(hidden, context) = self.lstm(result, (hidden,enc_context))
        result = self.lin(result[0])
        result = self.softmax(result)
        return result, (hidden, enc_context)
    
    def get_init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=self.device)
    
    def get_init_context(self):
        return torch.zeros(1, 1, self.hidden_dim, device=self.device)
