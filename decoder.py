# This Decoder class represents decoder part of the LSTM network. It will 
# generate a command based on the scenario input from the Encoder.

import torch
import torch.nn as nn

import config

class Decoder(nn.Module):
    # Initializes instance variables by calling pytorch functions
    def __init__(self, vocab_dim, embed_dim, hidden_dim):
        super(Decoder, self).__init__()

        # Initialize dimensions
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.device = config.device

        # Initialize lookup table to store indices to word embeddings
        self.embedding = nn.Embedding(self.vocab_dim, self.embed_dim)\
                            .to(self.device)
        # Initialize LSTM RNN
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim).to(self.device)
        # Initialize a linear transformation function
        self.lin = nn.Linear(self.hidden_dim, self.vocab_dim).to(self.device)
        # Initialize the softmax function as the activation function
        self.softmax = nn.Softmax(dim=1).to(self.device)
        
    # Run the decoder for one step. Gets the hidden state and encoder context
    # as input and finds the embedding for the given word. By passing this into
    # the LSTM, the linear function, and softmax, the result is returned.
    def forward(self, word, state):
        hidden = state[0].view(1, 1, -1)
        enc_context = state[1].view(1, 1, -1)
        result = self.embedding(word)
        result = result.view(1, 1, self.embed_dim)
        result,(hidden, context) = self.lstm(result, (hidden,enc_context))
        result = self.lin(result[0])
        result = self.softmax(result)
        return result, (hidden, enc_context)
    
    # Initialize the hidden state to a tensor of all zeroes
    def get_init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=self.device)
    
    # Initialize the encoder context to a tensor of all zeroes
    def get_init_context(self):
        return torch.zeros(1, 1, self.hidden_dim, device=self.device)
