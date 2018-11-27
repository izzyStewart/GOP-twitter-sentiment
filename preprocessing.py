# Importing python general modules needed for preprocessing.
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from collections import Counter

# Import Keras modules needed for preprocessing.
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split


def read_csv(file, col):
    """Function to read in csv files and turn them into arrays.
    """
    df = pd.read_csv(file)
    array = df[col].values
    return array

def text_to_words(array):
    """Function takes an array of text strings and turns them into lists of words as individual items.
    """
    for i in range(len(array)):
        array[i] = text_to_word_sequence(array[i])
    return array

def return_freq_list(all_words):
    """Function created to read a list of words and return a new list of words that appear more than once.
    """  
    counts = Counter(all_words)
    out_list = [i for i in counts if counts[i]>1]
    return out_list

def encodes(wordlist, tweets):
    """Function created to match word in tweet with index number of word in word list.
    """  
    encoding = []
    for word in tweets:
        for i, w in enumerate(wordlist):
            if(w == word):
                encoding.append(i)
    return encoding


class FormatLabels(object):
    """Class takes the array of class labels and turns into into a tensor suitable for a neural network.
    """
    
    def __init__(self, sentiment):
        """Initialisaing class.
        """
        self.sentiment = sentiment

    def text_to_num(self):
        """Function turns words into assigned number. 
        """
        labels = sum(self.sentiment, [])
        labels = [w.replace('neutral', '1') for w in labels]
        labels = [w.replace('positive', '2') for w in labels]
        labels = [w.replace('negative', '0') for w in labels]
        labels = np.array(labels)
        labels = labels.astype(int)
        return labels
        
    def one_hot(self, sentiment_labels):
        """Function one hot encodes numbers from list into correct tensor format. 
        """
        one_hot_labels = to_categorical(sentiment_labels)
        return one_hot_labels


class CreateWordList(object):
    """Class creates list of words from given text.
    """
    
    def __init__(self):
        """Initialising class.
        """ 

    def word_list(self, tweet_list):
        """Functiion takes an array of words as input and outputs a list of word that appear more than once.
        """
        all_words = sum(tweet_list, []) 
        reduced_result = return_freq_list(all_words)
        words = set(reduced_result)
        word_list = list(words)
        return word_list

    
class FormatTweetList(object):
    """Class creates vectorised tensor object from array of words.
    """
    
    def __init__(self):
        """Initialising class.
        """ 
        
    def tweet_encode(self, tweets, words):
        """Loops through tweets in array and applies the encodes function.
        """
        for i in range(len(tweets)):
            tweets[i] = encodes(words, tweets[i])
        return tweets 
   
    def vectorize_sequences(self, sequences, dimension = 7995):
        """Vectorizes numbers in each list item and returns in tensor format.
        """
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results
  

    