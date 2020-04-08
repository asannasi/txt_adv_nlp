# Global settings

# Set file names
data_file = "./data/data.txt"
verb_file = "./data/verbs.txt"
noun_file = "./data/nouns.txt"
prepos_file = "./data/prepos.txt"

# Set the start symbol, end symbol, and unknown symbol
SOS = "1"
EOS = "2"
UNK = "3"

# Function to get cuda device for pytorch
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
