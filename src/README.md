# Description of Source Files

## Overview for [Main Method](src/main.py)

0. Configure [config.py](src/config.py)
1. Collect data from Textworld with the [scraper script](scripts/scraper.py). Now there will be 4 text files with nouns, verbs, prepositions, and description-answer pairs, respectively. Make a training and test set by splitting data.txt into two files.
2. Parse the text files into a training set and a corpus using [make_training_game_data](src/data.py). Now we have a Data class instance
3. Initialize an [Encoder](src/encoder.py)
4. Initialize a [Decoder](src/decoder.py)
5. Initialize a [Trainer](src/trainer.py) from the training set, encoder, and decoder and set hyperparameters
6. Train the model by calling the trainer's train method for the desired number of cycles
7. Initialize an [Evaluator](src/evaluator.py) using the trained model in Trainer
8. Create the test set using [make_test_game_data](src/data.py)
9. Run the trained model on the test set with the evaluator's evaluate method
10. Display the trained model's loss function as a matplotlib graph
11. Repeat Steps 3-10 for different hyperparameter configurations

## Detailed Explanation

My neural network takes each word in the description and then generates a list of words as the answer, with the hope being that by inputting the answer, a player could win the game. Therefore, my model is an Encoder-Decoder LSTM network. Using an LSTM makes sense since the model should remember that it already outputted a certain word before so that it can generate meaningful phrases. 

### Encoder

The Encoder class embeds the word and then inputs it into the encoder’s LSTM along with the previous hidden vector and previous context vector. 

### Decoder

For the decoder, it takes the encoder’s final state as both the context vector and the initial hidden vector and inputs them into its LSTM along with the embedding of the previous word. No teacher forcing. The output of the LSTM is then fed into nn.Linear, which does a linear transformation to get a vector that is the size of the corpus. Then I feed this into a nn.Softmax to get a vector the size of the corpus with values between 0 and 1 at each index. These values represent the probabilities of each word being the word for this word vector prediction. I then take the index with the max probability as the word that is predicted. The target vector is a one-hot vector with a 1 at the index of the word that the answer has in the corpus. The optimizer I used is the stochastic gradient descent in pytorch. 

### Trainer

Set hyperparameters in the encoder-decoder network. It currently can set the encoder type, decoder type, the embedding dimension, the hidden state dimension, the learning rate, the loss function, and the loss factor, which is how much the loss is weighted. The Trainer class also has the optimizers for the encoder and decoder and will update their weights after finding the gradients. This update occurs after every pair of descriptions and answers.

### Evaluator

To evaluate my model, I defined an Evaluator class to evaluate the encoder-decoder network in the Trainer after training. The Evaluator class inputs each description into the trained encoder, gives the final state to the decoder, and then compares the predicted words to the answer. A success is counted if the predicted word is in the set of words in the answer, so order does not matter in this score. The number of successes is divided by the number of words in the answer to get a ratio of accuracy.
