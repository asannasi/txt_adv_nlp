# This class runs the LSTM network on the training set

import torch
import torch.nn as nn
import torch.optim as optim

import config

class Trainer:
    # Initialize instance variables
    def __init__(self, encoder, decoder, vocab_dim, embed_dim, hidden_dim,\
                    learning_rate, loss, loss_factor, corpus):
        # Set dimensions
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Setup network
        self.encoder = encoder
        self.decoder = decoder
        self.enc_optim = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.dec_optim = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        # Initialize learning rate and loss function and loss factor
        self.learning_rate = learning_rate
        self.loss_name = loss
        if(loss == "cross_entropy"):
            self.loss = nn.CrossEntropyLoss()
        elif(loss == "bce"):
            self.loss = nn.BCELoss()
        else:
            # Use xentropy
            self.loss = 0
        self.loss_factor = loss_factor

        self.corpus = corpus
        self.device = config.device
        self.curr_losses = [] # gives result of loss function
    
    # Train the encoder and decoder on the description-answer pair.
    # Returns the prediction from the decoder
    def train_pair(self, desc, ans):
        enc_output, enc_state = self.encoder.run(self.corpus, desc)
        predictions = self.train_decoder(ans,(enc_output, enc_output))
        return predictions
    
    # Calculate the loss based on the prediction made by the encoder-decoder
    def calc_loss(self, predictions, answers):
        L = torch.tensor(0,dtype=torch.float32)
        ans = answers.split(" ")
        for i in range(len(predictions)):
            # Calculate loss based on loss function
            if (self.loss_name == "cross_entropy"):
                obv = torch.tensor([self.corpus.index(ans[i])], \
                        dtype=torch.long, device=self.device)
                L += self.loss(predictions[i], obv)*self.loss_factor
            elif(self.loss_name == "bce"):
                obv = torch.zeros(1, len(self.corpus), dtype=torch.float32,\
                        device=self.device)
                obv[0][self.corpus.index(ans[i])] = 1
                L += self.loss(predictions[i], obv)*self.loss_factor
            else:
                obv = torch.zeros(len(self.corpus), dtype=torch.float32,\
                        device=self.device)
                obv[self.corpus.index(ans[i])] = 1
                L += self.xentropy_cost(obv, predictions[i][0])*self.loss_factor
        self.curr_losses.append(L)
        return L
    
    # Optimizes the gradient for the encoder and decoder
    def gradient(self, L):
        L.backward()
        self.enc_optim.step()
        self.dec_optim.step()
        
    # Train on all the descriptions and answers given
    def train(self, desc_list, ans_list):
        losses = []
        for i in range(len(desc_list)):
            self.enc_optim.zero_grad()
            self.dec_optim.zero_grad()
            predictions = self.train_pair(desc_list[i], ans_list[i])
            L = self.calc_loss(predictions, ans_list[i])
            L.backward()
            self.enc_optim.step()
            self.dec_optim.step()
        return self
    
    # Displays a graph of the losses made for predictions
    def graph_losses():
        plt.figure()
        plt.plot(self.curr_losses)
        plt.show()

    # Trains the decoder and gets predictions made
    def train_decoder(self, ans, encoder_state):
        hidden = encoder_state[0]
        context = encoder_state[1]
        predictions = []
        prev_word = torch.tensor(self.corpus.index("1"), device=self.device)
        for i in range(1,len(ans.split(" "))):
            input = torch.tensor([prev_word], dtype = torch.long,\
                    device=self.device)
            prev_state = (hidden, context)
            output, (hidden, context) = self.decoder.forward(input, prev_state)
            prev_word = torch.argmax(output[0], 0)
            predictions.append(output)
        return predictions

    # Defines a loss function based on
    #https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/10
    def xentropy_cost(self,x_target, x_pred):
        assert x_target.size() == x_pred.size()
        "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
        logged_x_pred = torch.log(x_pred)
        cost_value = -torch.sum(x_target * logged_x_pred)
        return cost_value
