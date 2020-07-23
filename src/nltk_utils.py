import nltk
import numpy as np
# Execute only the first time
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
        Split sentence into array of words/tokens.
        A token can be a word or punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
        Stemming in other words is find the root form of the word
        Examples:
            words = ["organize", "organizes", "organizing"]
            words = [stem(w) for w in words]
            -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
        Return bag of words array:
            1 for each known word that exists in the sentence
            0 otherwise
        Example:
            sentence    = ["hello", "how", "are", "you"]
            words       = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
            bag         = [  0,     1,     0,    1,     0,      0,       0  ]
    """
    # Stem each word
    sentence_words = [stem(w) for w in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1.0

    return bag
