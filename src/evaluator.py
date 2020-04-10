# This class will evaluate the test set and check the accuracy of the results.

import torch

import config

class Evaluator:
    # Initialize instance variables using the given trained Trainer instance
    def __init__(self, trainer):
        self.trainer = trainer
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.corpus = trainer.corpus
        self.device = config.device
        self.iterations = 10 # How many times to run the decoder on inputs
    
    # This function will run the encoder-decoder network on the new description
    # and return the predicted commands
    def predict_pair(self, desc):
        enc_output, enc_state = self.encoder.run(self.corpus, desc)
        predictions = self.run_decoder(self.corpus, enc_state, self.iterations)
        return predictions
    
    # This function will turn the predicted commands from one-hot vectors to 
    # the corresponding words in the corpus
    def get_words(self, predictions):
        words = []
        for i in range(len(predictions)):
            index = torch.argmax(predictions[i][0],0)
            word = self.corpus[index]
            words.append(word)
        return words
    
    # This function will compare the predicted command to the actual command
    # and return the ratio of correct commands
    def evaluate_pair(self, desc, ans):
        ans = ans.split(" ")
        predictions = self.predict_pair(desc)
        words = self.get_words(predictions)
        print("Description: ", desc)
        print("Answer: ", ans)
        print("Prediction:", words)
        counter = 0
        for i in range(len(ans)):
            if ans[i] in words:
                counter += 1
        return (counter,i)
            
    # This function will run all the descriptions and answer pairs in the test
    # data and return the number of correct commands predicted
    def evaluate(self, test_data):
        desc_list = test_data.descriptions
        ans_list = test_data.answers
        sum = 0
        successes = 0
        for i in range(len(desc_list)):
            (success, total) = self.evaluate_pair(desc_list[i], ans_list[i])
            sum += total
            successes += success
        return successes/sum

    # This function will run the decoder by keeping track of the encoder state
    # and iterating each word 
    def run_decoder(self, corpus, encoder_state, iterations):
        state = encoder_state
        hidden = state[0]
        context = state[1]
        # Skip the start symbol since this is present every time
        prev_word = torch.tensor(corpus.index("1"), device=self.device)

        predictions = []
        first = torch.zeros([1, len(corpus)], device=self.device)
        first[0][prev_word] = 1
        predictions.append(first)
        
        for i in range(0, iterations):
            input = torch.tensor([prev_word], dtype = torch.long,\
                    device=self.device)
            prev_state = (hidden, context)
            output, (hidden, context) = self.decoder.forward(input, prev_state)
            prev_word = torch.argmax(output[0], 0)
            predictions.append(output)
            
        return predictions
