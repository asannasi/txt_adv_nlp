# Text Adventure Game Parser

## Foreword
This is a final project from Fall 2018 that I proposed and developed based on a research paper to learn NLP. It used to be a 700-line Jupyter notebook, but now it's been refactored for my Github. It showcases how I simplified a complex problem and dealt with coding a large system. This project also shows how I handled working with a open-source API where the documentation did not match the newest release.

## Project Description
The goal of my project was to design a neural network that could play text adventure games. Text adventure games are games where a description of what is happening is provided via text. For example, a starting scenario could be “You are trapped in a room. There is a key on the table.” To progress in the game, the player would have to input commands such as “take key” or “unlock door”. Since all gameplay is done via text, I thought it would be the perfect way to learn about natural language processing.

Instead of creating a fully autonomous agent that could play text adenture games all by itself, I simplfied my goal into parsing the instructions scraped from the generated game and predicting the correct commands. I viewed this as a translation problem, so I decided to apply an encoder-decoder neural network.

## Project Implementation Details

Can be found [here](src/README.md).

## Dependencies

* Microsoft TextWorld: A open-source engine for generating text adventure game scenarios by Microsoft Research. I used the version available on Github on 12/18/2018. This is used only by the scraping script.
* Pytorch: Open source machine learning library
* Matplotlib: Used to compare hyperparameter effects on the loss functions by viewing graphs.

## How to run

Run the scraper by generating games using TextWorld and storing the data in text files.

```
python3 src/collect.py
```

Run the neural network with multiple hyperparameter configurations, which will produce graphs.

```
python3 src/main.py
```

## Results

After trying numerous hyperparameters for training, the best accuracy I achieved was 38.54%. All my results can be found in the graphs folder under the images folder, where it shows loss for each cycle.

| Hyperparameter | Value |
| --- | --- |
| Embedding Dimension | 100 |
| Hidden State Dimension | 1000 |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy |
| Loss Factor | 10000 |
| Training Cycles | 100 |

<a href="url"><img src="https://github.com/asannasi/txt_adv_nlp/blob/master/images/graphs/best.png" align="center" height="300" width="500" ></a>

## Sources
* https://arxiv.org/pdf/1705.05637.pdf
* Pytorch documentation
* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://bastings.github.io/annotated_encoder_decoder/
* https://github.com/Microsoft/TextWorld
* https://arxiv.org/pdf/1806.11532.pdf
* https://textworld-docs.maluuba.com/textworld.generator.game.html
