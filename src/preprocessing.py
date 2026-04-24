#================================================================================
# A Python library containing the implementation of various utility functions
# relevant to preprocessing text from Tweets
#================================================================================


"""Preprocessing"""
import re, string
import emoji
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions
from collections import Counter
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

NORMALIZE_MAP = {}

STOP_WORDS = set(stopwords.words('english'))
# I noticed that the lemmatizer
# converted the 'us' to 'u'
# so I did an update
STOP_WORDS.update({"u", "us", "dm", "fleet", "fleek"})

NORMALIZE_MAP = {}

SLANG_MAP = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "b": "be",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
}

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """ Cleans and normalizes raw tweet text for sentiment analysis.

    Performs the following steps:
        1. Converts emojis to words
        2. Removes URLs, @mentions, RT markers, and hashtags
        3. Expands contractions
        4. Lowercases all text
        5. Removes punctuation and digits
        6. Normalizes common Twitter slang

    Args:
        text (str): A raw tweet string.

    Returns:
        str: The cleaned and normalized tweet text.

    """

    # Convert emojis to words
    text = emoji.demojize(text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove RT (retweet) markers at the start or anywhere in text
    text = re.sub(r"(^RT\s+)|(\bRT\b)", "", text)
    # Split camelCase hashtags into separate words and remove the # symbol
    # e.g. #flightCancelled → flight Cancelled
    text = re.sub(r"#\w+", "", text)
    # text = re.sub(r"#(\w+)", lambda m: re.sub(r'([a-z])([A-Z])', r'\1 \2', m.group(1)), text)
    # Expand contractions (can't → cannot, won't → will not)
    text = contractions.fix(text)
    # Convert everything to lowercase
    text = text.lower()
    # Remove all punctuation characters (!, ?, ., :, etc.)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove any remaining digits
    text = re.sub(r"\d+", "", text)
    # Collapse multiple spaces into one and strip leading/trailing spaces
    # Keep only letters and underscores
    # underscores kept so emoji words like angry_face stay intact
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [SLANG_MAP.get(w, w) for w in words]
    words = re.findall(r"[a-z_]+", " ".join(words))  # ← use the updated words
    return " ".join(words)

def tokenize_and_clean(text):
    """Tokenize a document and remove stop words and punctuation

    Args:
        text(str): A raw string

    Returns:
        List: A list containing clean tokens of a document
    """

    if isinstance(text, str):
        tokens = word_tokenize(text)
    else:
        tokens = text

    tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def get_wordnet_pos(tag):
    """Helper function that returns the WordNet tag.

    Args:
        tag(wordnet.tag): A wordnet tag.
    Returns:
        str: wordnet.Tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def pos_tag_and_lemmatize(text):
    """
    Performs POS-guided lemmatization using spaCy for tagging
    and WordNet for lemmatization.

    Args:
        text (str or list): A cleaned tweet string or list of tokens.

    Returns:
        list: A list of lemmatized tokens
    """

    if isinstance(text, list):
        text = " ".join(text)
    pos_doc = nlp(text)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.text, get_wordnet_pos(token.tag_)) for token in pos_doc]

    return tokens

def normalize_tokens(tokens):
    """Normalize explicitly certain words
    The words must be defined in a dictionary 'NORMALIZED_MAP

    Args:
        tokens(str): A word string

    Returns:
        List: of the normalized words
    """

    return [NORMALIZE_MAP.get(t, t) for t in tokens]

def remove_consecutive_dupes(tokens):
    """Remove consecutive duplicates from a list or tokens

    Args:
        token(str): A word string.

    Returns:
        List: A cleaned list of tokens.
    """
    return [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i-1]]