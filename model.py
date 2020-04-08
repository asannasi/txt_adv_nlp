import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from data import *
from encoder import Encoder
from decoder import Decoder
from trainer import Trainer
from evaluator import Evaluator

def hyperparams():
    print("This is the baseline ratio I am going to use")
    
    print("The vocab dim is length of the corpus.")
    print("The embedding dim is 30.")
    print("The hidden dim is 100.")
    print("The learning rate is 0.001.")
    print("The loss function is cross entropy.")
    print("The loss is multiplied by a factor of 10000.")
    print("Training is run for 2 cycles.")

    train_data = make_training_game_data(config.data_file, config.noun_file,\
        config.verb_file, config.prepos_file)
    c = train_data.corpus
    d = train_data.descriptions
    a = train_data.answers
    enc = Encoder(len(c), 100, 100)
    dec = Decoder(len(c), 100, 100)

    # Trainer(encoder_type, decoder_type, vocab_dim, embed_dim, hidden_dim, 
    #         learning_rate, loss function, loss factor, corpus)
    t = Trainer(enc, dec, len(c), 30, 100, 0.001, \
            "cross_entropy", 10000, train_data.corpus)

    for i in range(0, 2):
        t = t.train(d, a)
        
    e = Evaluator(t)
    test_data = make_test_game_data(train_data.corpus, "./data/data2.txt")
    ratio = e.evaluate(test_data)
    plt.figure()
    plt.plot(t.curr_losses)
    plt.show()
    print("RATIO: ", ratio)
    print("*****")
    
    print("Now with 50 cycles")
    enc = Encoder(len(c), 100, 100)
    dec = Decoder(len(c), 100, 100)
    t2 = Trainer(enc, dec, len(c), 30, 100, 0.001, \
            "cross_entropy", 10000, train_data.corpus)
    for i in range(0, 50):
        t2 = t2.train(d, a)
        
    e = Evaluator(t2)
    test_data = make_test_game_data(train_data.corpus, "./data/data2.txt")
    ratio2 = e.evaluate(test_data)
    plt.figure()
    plt.plot(t2.curr_losses)
    plt.show()
    print("RATIO: ", ratio2)
    print("*****")
    
    print("Now with a self-defined loss function of xentropy_cost\
            for one hot vectors")
    enc = Encoder(len(c), 100, 100)
    dec = Decoder(len(c), 100, 100)
    t2 = Trainer(enc, dec, len(c), 30, 100, 0.001, "x", \
            10000, train_data.corpus)
    for i in range(0, 50):
        t2 = t2.train(d, a)
        
    e = Evaluator(t2)
    test_data = make_test_game_data(train_data.corpus, "data2.txt")
    ratio2 = e.evaluate(test_data)
    plt.figure()
    plt.plot(t2.curr_losses)
    plt.show()
    print("RATIO: ", ratio2)
    print("*****")
    
    print("Now with the loss function BCE")
    enc = Encoder(len(c), 100, 100)
    dec = Decoder(len(c), 100, 100)
    t2 = Trainer(enc, dec, len(c), 30, 100, 0.001, "bce", \
            10000, train_data.corpus)
    for i in range(0, 50):
        t2 = t2.train(d, a)
        
    e = Evaluator(t2)
    test_data = make_test_game_data(train_data.corpus, "data2.txt")
    ratio2 = e.evaluate(test_data)
    plt.figure()
    plt.plot(t2.curr_losses)
    plt.show()
    print("RATIO: ", ratio2)
    print("*****")
    
    print("Now with the loss function BCE and 100 cycles")
    t2 = Trainer(1, 1, len(c), 30, 100, 0.001, "bce", 10000, train_data.corpus)
    for i in range(0, 100):
        t2 = t2.train(d, a)
        
    e = Evaluator(t2)
    test_data = make_test_game_data(train_data.corpus, "data2.txt")
    ratio2 = e.evaluate(test_data)
    plt.figure()
    plt.plot(t2.curr_losses)
    plt.show()
    print("RATIO: ", ratio2)
    print("*****")
    
    print("Now with the loss function BCE and 100 cycles and learning rate of 0.0001")
    t2 = Trainer(1, 1, len(c), 30, 100, 0.0001, "bce", 10000, train_data.corpus)
    for i in range(0, 100):
        t2 = t2.train(d, a)
        
    e = Evaluator(t2)
    test_data = make_test_game_data(train_data.corpus, "data2.txt")
    ratio2 = e.evaluate(test_data)
    plt.figure()
    plt.plot(t2.curr_losses)
    plt.show()
    print("RATIO: ", ratio2)
    print("*****")
    
    print("Now with the loss function BCE and 100 cycles and learning rate of 0.0001 and loss factor of 100000")
    t2 = Trainer(1, 1, len(c), 30, 100, 0.001, "bce", 100000, train_data.corpus)
    for i in range(0, 100):
        t2 = t2.train(d, a)
        
    e = Evaluator(t2)
    test_data = make_test_game_data(train_data.corpus, "data2.txt")
    ratio2 = e.evaluate(test_data)
    plt.figure()
    plt.plot(t2.curr_losses)
    plt.show()
    print("RATIO: ", ratio2)
    print("*****")
    
    print("Now embedding dim is 100, hidden dim is 1000, learning rate is 0.001, loss function is bce, the loss is multiplied by a factor of 10000, and training for 100 cycles")
    t2 = Trainer(1, 1, len(c), 100, 1000, 0.001, "bce", 10000, train_data.corpus)
    for i in range(0, 100):
        t2 = t2.train(d, a)
        
    e = Evaluator(t2)
    test_data = make_test_game_data(train_data.corpus, "data2.txt")
    ratio2 = e.evaluate(test_data)
    plt.figure()
    plt.plot(t2.curr_losses)
    plt.show()
    print("RATIO: ", ratio2)
    print("*****")
    
print("Starting...")
hyperparams()

print("Starting")
def best():
    train_data = make_training_game_data("data4.txt", "nouns4.txt", "verbs4.txt", "prepos4.txt")
    c = train_data.corpus
    d = train_data.descriptions
    a = train_data.answers
    print("Now with the loss function BCE and 100 cycles")
    t2 = Trainer(1, 1, len(c), 30, 100, 0.001, "bce", 10000, train_data.corpus)
    for i in range(0, 100):
        t2 = t2.train(d, a)
        
    e = Evaluator(t2)
    test_data = make_test_game_data(train_data.corpus, "data2.txt")
    ratio2 = e.evaluate(test_data)
    plt.figure()
    plt.plot(t2.curr_losses)
    plt.show()
    print("RATIO: ", ratio2)
    print("*****")
best()
