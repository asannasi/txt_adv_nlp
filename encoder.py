import torch
import torch.nn as nn

import config

class Encoder(nn.Module):
    def __init__(self, vocab_dim, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = config.device
        self.embedding = nn.Embedding(self.vocab_dim, self.embed_dim)\
                            .to(self.device)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim).to(self.device)
        
    def forward(self, word, state):
        prev_h = state[0].view(1, 1, -1)
        prev_c = state[1].view(1, 1, -1)
        result = self.embedding(word)
        result = result.view(1, 1, self.embed_dim)
        result,(hidden, context) = self.lstm(result, (prev_h, prev_c))
        return result, (hidden, context)
    
    def get_init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim,device=self.device)
    
    def get_init_context(self):
        return torch.zeros(1, 1, self.hidden_dim,device=self.device)
    
    def run(self, corpus, desc):    
        hidden = self.get_init_hidden()
        context = self.get_init_context()
        output = 0
        
        for word in desc.split(" "):
            if word not in corpus:
                ind = corpus.index(config.UNK) #UNK symbol
            else:
                ind = corpus.index(word)
            input = torch.tensor([ind], dtype = torch.long, device=self.device)
            prev_state = (hidden, context)
            output, (hidden, context) = self.forward(input, prev_state)
        
        return output, (hidden, context)
