# This python script scrapes text adventure game scenarios
# from the Microsoft TextWorld engine and stores features
# like verbs, nouns, and prepositions in the data. These
# are stored into the respective 4 files.

import textworld # Version from 12/8/18
import numpy as np
import os

import config

# Game settings
NUM_GAMES = 10000
GAME_DIR = "./gen_games/" 
QUEST_BREADTH = 50
NB_OBJECTS = 50
QUEST_LENGTH = 50

def main():
    data_file = open(config.data_file, "a")
    verb_file = open(config.verbs_file, "a")
    noun_file = open(config.nouns_file, "a")
    prepos_file = open(config.prepos_file, "a")

    # Create a new game for every iteration
    for game in range(NUM_GAMES):
        # Clear folder of previous game
        test = os.listdir(GAME_DIR)
        for item in test:
            if item.endswith(".json") or item.endswith(".ni") or\
                 item.endswith(".ulx"):
                os.remove(os.path.join(GAME_DIR, item))

        # Set game options
        options = textworld.GameOptions()
        options.seeds = np.random.randint(0, NUM_GAMES)
        options.quest_breadth = QUEST_BREADTH
        options.nb_objects = NB_OBJECTS
        options.quest_length = QUEST_LENGTH

        # Initialize textworld
        game_file, _ = textworld.make(options)
        env = textworld.start(game_file)

        # Set flags
        compute_intermediate_reward()
        env.activate_state_tracking()
        game_state = env.reset()
        env.compute_intermediate_reward()
        env.activate_state_tracking()

        # Get the objective generated in natural language 
        # along with the correct commands a player should input.
        # Store the string result into the data file.
        objective = game_state.objective
        separator = "//"
        print(game_state.policy_commands)
        answer = ' '.join(game_state.policy_commands)
        data = objective + separator + answer + '\n'
        data_file.write(data)

        # Get all verbs and store them in the verbs file.
        v = ' '.join(game_state.verbs)
        v = v + '\n'
        verb_file.write(v)

        # Get all nouns and store them in the nouns file.
        n = ' '.join(game_state.entities)
        n = n + '\n'
        noun_file.write(n)

        # Get all prepositions and store them in its file.
        p = ' '.join(game_state.command_templates)
        p_split = p.split(' ')
        for word in p_split:
            if word[0] == '{':
                p_split.remove(word)
        p = ' '.join(p_split)
        p = p + '\n'
        prepos_file.write(p)

if __name__ == "__main__":
    main()
