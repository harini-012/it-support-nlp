import os

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)

from app.nlp import NLPPreprocessor


ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_PATH = os.path.join(
    ROOT,
    "data",
    "tickets.csv"
)


MODEL_DIR = os.path.join(
    ROOT,
    "models"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ---------------------------------------
# Load dataset
# ---------------------------------------

df = pd.read_csv(
    DATA_PATH
)


df["text"] = (
    df["title"].fillna("")
    + " "
    + df["description"].fillna("")
)


# ---------------------------------------
# NLP
# ---------------------------------------

processor = NLPPreprocessor()


df["clean_text"] = df["text"].apply(
    processor.clean_text
)


# ---------------------------------------
# Model factory
# ---------------------------------------

def create_model():

    return Pipeline([

        (
            "tfidf",

            TfidfVectorizer(

                ngram_range=(1, 2),

                sublinear_tf=True,

                max_features=10000
            )
        ),

        (
            "classifier",

            LogisticRegression(

                max_iter=3000
            )
        )
    ])


# ---------------------------------------
# Category model
# ---------------------------------------

category_model = create_model()


category_model.fit(

    df["clean_text"],

    df["category"]
)


# ---------------------------------------
# Priority model
# ---------------------------------------

priority_model = create_model()


priority_model.fit(

    df["clean_text"],

    df["priority"]
)


# ---------------------------------------
# Save
# ---------------------------------------

joblib.dump(

    category_model,

    os.path.join(
        MODEL_DIR,
        "category_model.pkl"
    )
)


joblib.dump(

    priority_model,

    os.path.join(
        MODEL_DIR,
        "priority_model.pkl"
    )
)


joblib.dump(

    processor,

    os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )
)


print()
print("=" * 60)

print(
    "MODELS TRAINED SUCCESSFULLY"
)

print("=" * 60)

print(
    "Category model:"
)

print(
    "models/category_model.pkl"
)

print(
    "Priority model:"
)

print(
    "models/priority_model.pkl"
)