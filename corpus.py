# This is a collection of helper functions for
# parsing the files from collect.py into a corpus

# Helper function to parse the nouns, verbs, 
# and prepositions file for corpus words
def parse_word_file(file):
    word_list = []
    f = open(file, "r")
    for group in f.read().splitlines():
        for word in group.split(' '):
            word_list.append(word.lower())
    f.close()
    return set(word_list)

# Helper function to parse the data file for corpus words
def parse_data_file_for_words(file):
    word_list = []
    f = open(file, "r")
    for group in f.read().splitlines():
        for word in group.split(' '):
            word = word.split("//")
            word[0] = ''.join(filter(str.isalpha, word[0]))
            word_list.append(word[0].lower())
            if (type(word) == list and len(word)>1):
                word[1] = ''.join(filter(str.isalpha, word[1]))
                word_list.append(word[1].lower())
    f.close()
    return set(word_list)

# Get the description and commands mapping
def parse_data_file_for_pairs(file):
    descriptions = []
    answers = []
    f = open(file, "r")
    # Iterate through each description and command pair
    # and store them into separate lists.
    for group in f.read().splitlines():
        group = group.split("//")

        #https://stackoverflow.com/questions/2779453/
        import re
        pattern = re.compile('([^\s\w]|_)+')
        
        group[0] = pattern.sub('', group[0])
        descriptions.append(group[0].lower())
        group[1] = pattern.sub('', group[1])
        answers.append(group[1].lower())
    f.close()

    # Change each corpus into ASCII
    descriptions = corpusToAscii(descriptions)
    answers = corpusToAscii(answers)

    return descriptions, answers

# Parse given files for corpus
def gen_corpus(data_file, noun_file, verb_file, prepos_file):
    return set(parse_word_file(noun_file) | parse_word_file(verb_file) | 
                parse_word_file(prepos_file) | \
                parse_data_file_for_words(data_file))

# This function changes unicode to ASCII
#https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
def unicodeToAscii(s):
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# This function changes each word in a corpus into ASCII
def corpusToAscii(corpus):
    new_corpus = []
    for word in corpus:
        new_corpus.append(unicodeToAscii(word))
    return new_corpus

# Adds start and end symbols to corpus
def add_symbols(corpus, SOS, EOS, UNK):
    if(type(corpus) == set):
        corpus.add(SOS)
        corpus.add(EOS)
        corpus.add(UNK)
    else:
        corpus.append(SOS)
        corpus.append(EOS)
        corpus.append(UNK)
    return corpus

# Adds start symbols to data-command pairs
def add_starts(descriptions, answers, SOS):
    for i in range(len(descriptions)):
        descriptions[i] = SOS + ' ' + descriptions[i]
        answers[i] = SOS + ' ' + answers[i]
    return descriptions, answers

# Adds end symbols to data-command pairs
def add_ends(descriptions, answers, EOS):
    for i in range(len(descriptions)):
        descriptions[i] = descriptions[i] + ' ' + EOS
        answers[i] = answers[i] + ' ' + EOS
    return descriptions, answers
