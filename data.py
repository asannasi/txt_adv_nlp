# This program defines a class that holds the corpus and the description to
# answer pairs for game scenarios.

import re

import config
from corpus import *

# Organize description-answer pairs and corpus into one data structure
class Data:
    def __init__(self, corpus, descriptions, answers):
        self.descriptions = descriptions
        self.answers = answers
        self.corpus = list(corpus)

# Create training set from scraped data stored in files
def make_training_game_data(data_file, noun_file, verb_file, prepos_file):
    # Create corpus of words
    corpus = gen_corpus(data_file, noun_file, verb_file, prepos_file)
    # Add start and end symbol into corpus
    corpus = add_symbols(corpus, config.SOS, config.EOS, config.UNK)
    corpus = corpusToAscii(corpus)

    # Get the pairs of the instructions and the command list
    descriptions, answers = parse_data_file_for_pairs(data_file)
    descriptions, answers = add_starts(descriptions, answers, config.SOS)
    descriptions, answers = add_ends(descriptions, answers, config.EOS)

    # Store all the data processed into a class
    game_data = Data(corpus, descriptions, answers)
    return game_data

# Create test set from scraped data stored in files
def make_test_game_data(corpus, data_file):
    # Get the pairs of the instructions and the command list
    descriptions, answers = parse_data_file_for_pairs(data_file)
    # Add start and end symbol into corpus
    descriptions, answers = add_starts(descriptions, answers, config.SOS)
    descriptions, answers = add_ends(descriptions, answers, config.EOS)
    # Store all the data processed into a class
    game_data = Data(corpus, descriptions, answers)
    return game_data

# Combine the existing corpus Data instance with data from other files
def combine_corpus(t1, data_file, noun_file, verb_file, prepos_file):
    t2 = make_training_game_data(data_file, noun_file, verb_file, prepos_file)
    c = set(t1.corpus + t2.corpus)
    d = t1.descriptions + t2.descriptions
    a = t1.answers + t2.answers
    return Data(c, d, a)
