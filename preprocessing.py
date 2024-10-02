# imports
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce
import pandas as pd

# --------------------------------------------
def bag_of_words(T):
    """
    Converts string to bag of words representation
    Param T: list/iterable of strings
    Returns: list of list of words
    """
    # TODO: lower case missing?
    return [word_tokenize(t) for t in T]
    # Out: list of list of words

    no_punc = "".join(c if c.isalnum() else " " for c in T)
    return no_punc.lower().split()

#Deprecated
def extract_relevant_words(T):
    """
    Remove stop words
    """
    with open("resources/stopwords/english_stopwords.txt", "r") as f:
        stopwords = set(f.read().split())
    return [w for w in T if w not in stopwords]

def stemming(U):
    """
    Stemming - remove conjugations etc
    Param U: list/iterable of list/iterable of words
    Returns: list of strings with stemmed words
    """
    stemmer = PorterStemmer()

    return [reduce(lambda x, y: x + " " + stemmer.stem(y), u, "") for u in U]

    return [stemmer.stem(w) for w in U]

#if needed
def extract_features(V, binary=False):
    """
    Extract features via bag of words

    Param V: sequence of items of type string/byte
    Returns: Pandas dataframe of features
    """
    cv = CountVectorizer(strip_accents='ascii', lowercase=True, tokenizer=None, stop_words='english', binary=binary) # can make tokenizer use bag of words tokenizer to reduce time

    feature_matrix = cv.fit_transform(V)

    feature_array = feature_matrix.toarray()

    df = pd.DataFrame(data=feature_array, columns=cv.get_feature_names_out())

    return df, cv, feature_matrix, feature_array
    
    
    return stemming(extract_relevant_words(bag_of_words(V)))

#---------------------------------------------
# Scaling (choose one):
# Can also standardize and then normalize

# Standardization ==> ~N(u=0, s=1)
# Use if data follows normal distribution
# Better for algos that don't assume distributions e.g. k-nearest neighbors, neural nets
def standardize(W):
    """
    Perform Standardization to make all features have mean of 0 and sd/var of 1
    
    Returns transformed data and StandardScaler object
    """
    ss = StandardScaler()
    X = ss.fit_transform(W)
    return X, ss

# Normalization ==> [0,1]
# Default / Use if data does not follow normal distribution
# Better for algos that assume distributions e.g. linear regression
def normalize(W):
    """
    Perform Normalization to make all features have range of [0,1]
    
    Returns transformed data and MinMaxScaler object
    """
    mms = MinMaxScaler()
    X = mms.fit_transform(W)
    return X, mms



#--------------------------------------------
# Dimension Reduction (choose one):

def perform_PCA(W, n_components=0.999):
    """
    Perform PCA - reduce features
    Default: Return enough features to account for 99.9% of variance, can attempt 100 features or other values as desired.

    Returns transformed data and PCA object
    """

    pca = PCA(n_components=n_components)

    X= pca.fit_transform(W)

    return X, pca


#--------------------------------------------
if __name__ == '__main__':
    print(bag_of_words("SpiceJet to issue 6.4 crore warrants to promoters"))
    print(extract_relevant_words(bag_of_words("SpiceJet to issue 6.4 crore warrants to promoters")))
    print(stemming(extract_relevant_words(bag_of_words("SpiceJet to issue 6.4 crore warrants to promoters"))))
    print(extract_features("SpiceJet to issue 6.4 crore warrants to promoters"))
    # Un-Processed Data
    T = []
    
    exit()
