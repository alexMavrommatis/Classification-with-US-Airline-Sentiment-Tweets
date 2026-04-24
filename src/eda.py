'''Data'''
import pandas as pd
import numpy as np
'''EDA'''
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.probability import FreqDist
'''Visualization'''
import seaborn as sns
import matplotlib.pyplot as plt


def plot_horizontal_bar(df, x, y, title, xlabel="Frequency", ylabel="Word"):
    """Display a horizontal bar chart using seaborn.

    Args:
        df: DataFrame containing the data.
        x: Column name for the x-axis (bar lengths).
        y: Column name for the y-axis (bar labels).
        title: Chart title.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def get_top_ngrams(all_tokens, n, top=10):
    """Count the number of most frequent ngram tokens in a list of tokens.

    Args:
        all_tokens(List(str)): A list containing the tokens for counting.
        n(int): Then number of n grams to be counted.
        top(int): The the most frequent ngrams to be returned.

    Returns:
        Counter: A Counter set of the most frequent ngrams.
    """

    bag = list(ngrams(all_tokens, n))

    return Counter(bag).most_common(top)

def get_top_by_sentiment(df, sentiment_col, tokens_col, n_top=10, ngram=None):
    """Compute the most frequent words or n-grams per sentiment category.

    Args:
        df: DataFrame with tokenized text.
        sentiment_col: Name of the sentiment column.
        tokens_col: Name of the column containing token lists.
        n_top: Number of top items to return per sentiment.
        ngram: If None, count individual words. If int (e.g. 2), count n-grams of that size.

    Returns:
        dict: {sentiment: DataFrame with 'word' and 'count' columns}.
    """
    results = {}
    for sentiment in df[sentiment_col].unique():
        subset = df[tokens_col].loc[df[sentiment_col] == sentiment]
        all_tokens = [t for tokens in subset for t in tokens]
        if ngram:
            top = get_top_ngrams(all_tokens, n=ngram, top=n_top)
        else:
            top = FreqDist(all_tokens).most_common(n_top)
        results[sentiment] = pd.DataFrame(top, columns=['word', 'count'])
    return results


