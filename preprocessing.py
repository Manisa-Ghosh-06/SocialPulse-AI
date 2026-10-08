import re
import nltk
import os

NLTK_DATA="/tmp/nltk_data"
os.makedirs(NLTK_DATA,exist_ok=True)
nltk.data.path.insert(0,NLTK_DATA)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def setup_nltk():

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for path, resource in resources:

        try:
            nltk.data.find(path)

        except LookupError:
            nltk.download(resource, download_dir=NLTK_DATA,quiet=True)


setup_nltk()


stop_words = set(
    stopwords.words("english")
) - {
    "not",
    "no",
    "nor"
}


lemmatizer = WordNetLemmatizer()


def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def preprocess_text(text):

    text = clean_text(text)

    if not text:
        return ""

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)
