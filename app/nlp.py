import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class NLPPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = str(text).lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [
            token
            for token in tokens
            if token not in self.stop_words
        ]

        # Remove very short tokens
        tokens = [
            token
            for token in tokens
            if len(token) > 1
        ]

        # Lemmatization
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
        ]

        return " ".join(tokens)