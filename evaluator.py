import torch

import config

class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.corpus = trainer.corpus
        self.device = config.device
        
    def predict_pair(self, desc):
        enc_output, enc_state = self.encoder.run(self.corpus, desc)
        predictions = self.run_decoder(self.corpus, enc_state)
        return predictions
    
    def get_words(self, predictions):
        words = []
        for i in range(len(predictions)):
            index = torch.argmax(predictions[i][0],0)
            word = self.corpus[index]
            words.append(word)
        return words
    
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
            
    def evaluate(self, test_data):
        desc_list = test_data.descriptions
        ans_list = test_data.answers
        sum = 0
        successes = 0
        for i in range(len(desc_list)):
            (success, total) = self.evaluate_pair(desc_list[i], ans_list[i])
            sum += total
            successes += success
            
            #print progress
            #if(i % 1000 == 0):
                #print("Testing...", i/len(desc_list))
        return successes/sum

    def run_decoder(self, corpus, encoder_state):
        state = encoder_state
        hidden = state[0]
        context = state[1]
        prev_word = torch.tensor(corpus.index("1"), device=self.device)

        predictions = []
        first = torch.zeros([1, len(corpus)], device=self.device)
        first[0][prev_word] = 1
        predictions.append(first)
        
        for i in range(0, 10):
            input = torch.tensor([prev_word], dtype = torch.long, device=self.device)
            prev_state = (hidden, context)
            output, (hidden, context) = self.decoder.forward(input, prev_state)
            prev_word = torch.argmax(output[0], 0)
            predictions.append(output)
            
        return predictions
