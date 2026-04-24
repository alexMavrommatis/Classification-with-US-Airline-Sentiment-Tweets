# Airline Tweet Sentiment Classification with ANN

Classification of US airline tweets into **Negative**, **Neutral**, and **Positive** sentiment using a feed-forward neural network (ANN) built with PyTorch.

## Dataset

Twitter US Airline Sentiment dataset (~14,640 tweets).

## Pipeline

1. **Preprocessing** — removal of mentions, hashtags, URLs, punctuation, stopwords; lowercasing; contraction expansion; POS-based lemmatization.
2. **EDA** — sentiment distribution, word frequency analysis, bigram analysis per class.
3. **Vectorization** — TF-IDF (max 6,000 features, unigrams + bigrams).
4. **Baseline** — Logistic Regression.
5. **ANN** — 2-hidden-layer MLP (256 → 128), dropout 0.2, Adam optimizer, early stopping.
## Results

| Model               | Macro F1 |
|----------------------|----------|
| Logistic Regression  | 0.68     |
| ANN                  | 0.70     |

The ANN improves recall on the Neutral and Negative classes. Both models struggle with Neutral tweets due to vocabulary overlap with the Negative class.

## Project Structure

```
├── data/
│   └── 01_raw/Tweets.csv
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── ann_classification.py
│   └── ann_utility.py
├── notebooks/
│   └── main_notebook.ipynb
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch, scikit-learn, NLTK, spaCy, pandas, numpy, seaborn, matplotlib
Install with:

```bash
pip install torch scikit-learn nltk spacy pandas numpy seaborn matplotlib contractions emoji
```

## How to Run

Open and run the main notebook end-to-end. Ensure the dataset is placed under `data/01_raw/Tweets.csv`.