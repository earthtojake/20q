Instructions for setting up the 20Q bot:

The program runs on Python 2.7, modules are in requirements.txt: numpy==1.12.0, matplotlib==2.0.2

There are two modes and four datasets which you can use to interact with the bot:

MODES:
1. 'play': load a previously trained and optimised network.
2. 'crossvalidate': train a network for a dataset and then play with that network.

DATASETS:
1. 'micro': for very basic training. 5 animals + 5 questions, question limit = 4.
2. 'small': testing basic intelligence. 12 animals + 6 questions, question limit = 5.
3. 'medium': for testing smart question selection. 13 animals + 15 questions, question limit = 6.
4. 'big': the big kahuna... 58 animals + 28 questions, question limit = 13. 

To run the bot in either mode with a dataset:

python 20q.py <DATASET> <MODE>

e.g.
python 20q.py medium crossvalidate
python 20q.py big play

It is recommended that 'crossvalidate' mode only be used on datasets micro, small or medium (not big),
as training for 'big' takes 20-30 mins.

Github link: https://github.com/jakefitzgerald/20q