
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from encoder import Encoder
from decoder import Decoder
from data import *
from trainer import Trainer
from evaluator import Evaluator
import config

def encoder_test():
    c = ["hi", "bye", "no"]
    e = Encoder(len(c), 100, 100)
    param = e.parameters()
    optimi = optim.SGD(e.parameters(), lr=0.001)
    o1, (h1, c1) = e.run(c, "hi bye no")
    e.run(c, "hi bye no")
    loss = nn.CrossEntropyLoss()
    target = torch.tensor([3], dtype=torch.long, device=config.device)
    losses = []
    K=0
    for i in range(0, 500):
        o1, (h1, c1) = e.run(c, "hi bye no")
        K = loss(h1[0], target)*1
        K.backward(retain_graph=True)
        optimi.step()
        losses.append(K)
        print(K)
    plt.figure()
    plt.plot(losses)
    print(o1[0].size())
    print(o1[0][0])

    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()
    print(input)
    print(target)
    print(output)

    c = ["1","hi", "bye", "no"]
    dec = Decoder(len(c),100, 100)
    param = dec.parameters()
    optimi = optim.SGD(dec.parameters(), lr=0.001)
    ans = "1 hi bye no"
    encoder_state = (o1, o1)
    hidden = encoder_state[0]
    context = encoder_state[1]
    predictions = []
    prev_word = torch.tensor(c.index("1"), device=config.device)
    for i in range(1,len(ans.split(" "))):
        input = torch.tensor([prev_word], dtype = torch.long, device=config.device)
        prev_state = (hidden, context)
        output, (hidden, context) = dec.forward(input, prev_state)
        prev_word = torch.argmax(output[0], 0)
        predictions.append(output)
    pred = predictions
    print(pred)
    target = torch.zeros(len(c), dtype=torch.float32, device=config.device)
    target[2] = 1
    torch.log(pred[0][0])
    x_target = target
    x_pred = pred[0][0]
    logged_x_pred = torch.log(x_pred)
    cost_value = -torch.sum(x_target * logged_x_pred)
    M = cost_value * 1
    print(M)
    M.backward(retain_graph=True)
    optimi.step()

def train_test():
    td = make_training_game_data(config.data_file, config.noun_file,\
        config.verb_file, config.prepos_file)
    c = td.corpus
    d = td.descriptions
    a = td.answers
    enc = Encoder(len(c), 100, 100)
    dec = Decoder(len(c), 100, 100)
    
    t = Trainer(enc,dec, len(c), 30, 100, 0.001, "x", 100, c)
    for i in range(0, 300):
        t = t.train(d, a)

def train_test_cross():
    td = make_training_game_data(config.data_file, config.noun_file,\
        config.verb_file, config.prepos_file)
    c = td.corpus
    d = td.descriptions
    a = td.answers
    enc = Encoder(len(c), 100, 100)
    dec = Decoder(len(c), 100, 100)
    
    t = Trainer(enc, dec, len(c), 30, 100, 0.001, "cross_entropy", 10000, c)
    for i in range(0, 2):
        t = t.train(d, a)

    t = Trainer(enc,dec, len(c), 100, 100, 0.0000001, "cross_entropy", 100000, c)
    t.train(d,a)
    return t

def eval_test1(t):
    td = make_training_game_data(config.data_file, config.noun_file,\
        config.verb_file, config.prepos_file)
    e = Evaluator(t)
    p = e.evaluate_pair(td.descriptions[0], td.answers[0])
    p = e.evaluate_pair(td.descriptions[1], td.answers[0])
    print(td.descriptions[1])
    print(p)

def eval_test2(t):
    td = make_training_game_data(config.data_file, config.noun_file,\
        config.verb_file, config.prepos_file)
    test_data = make_test_game_data(td.corpus, "./data/data2.txt")
    e = Evaluator(t)
    o = e.evaluate(test_data)
    print(o)

def main():
    encoder_test()
    train_test()
    eval_trainer = train_test_cross()
    eval_test1(eval_trainer)
    eval_test2(eval_trainer)

if __name__ == "__main__":
    main()
