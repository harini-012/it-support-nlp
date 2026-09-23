import os

import joblib
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_DIR = os.path.join(
    ROOT,
    "models"
)


DATA_PATH = os.path.join(
    ROOT,
    "data",
    "tickets.csv"
)


# ---------------------------------------
# Load models
# ---------------------------------------

category_model = joblib.load(

    os.path.join(
        MODEL_DIR,
        "category_model.pkl"
    )
)


priority_model = joblib.load(

    os.path.join(
        MODEL_DIR,
        "priority_model.pkl"
    )
)


processor = joblib.load(

    os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )
)


# ---------------------------------------
# Load historical tickets
# ---------------------------------------

df = pd.read_csv(
    DATA_PATH
)


df["text"] = (

    df["title"].fillna("")

    + " "

    + df["description"].fillna("")
)


df["clean_text"] = df["text"].apply(
    processor.clean_text
)


# ---------------------------------------
# Similarity model
# ---------------------------------------

similarity_vectorizer = TfidfVectorizer(

    ngram_range=(1, 2),

    sublinear_tf=True
)


similarity_matrix = (

    similarity_vectorizer.fit_transform(

        df["clean_text"]
    )
)


# ---------------------------------------
# Team mapping
# ---------------------------------------

TEAM_MAPPING = {

    "Network":
        "Network Support",

    "Hardware":
        "Hardware Support",

    "Software":
        "Software Support",

    "Database":
        "Database Team",

    "Access":
        "Access Management",

    "Security":
        "Security Team",

    "Application":
        "Application Support",

    "Email":
        "Email Support",

    "Cloud":
        "Cloud Operations"
}


# ---------------------------------------
# Suggested solutions
# ---------------------------------------

SOLUTIONS = {

    "Network":
        "Check network connectivity, WiFi configuration, VPN settings and network logs.",

    "Hardware":
        "Check device hardware, drivers, cables, battery and physical components.",

    "Software":
        "Restart the application and check installation, updates and application logs.",

    "Database":
        "Check database connectivity, credentials, query performance and database logs.",

    "Access":
        "Verify account status, credentials and required permissions.",

    "Security":
        "Isolate the affected system and contact the security team immediately.",

    "Application":
        "Check application availability, logs and configuration.",

    "Email":
        "Check mailbox storage, email configuration and server connectivity.",

    "Cloud":
        "Check cloud service health, deployment status and configuration."
}


# ---------------------------------------
# Prediction
# ---------------------------------------

def predict_ticket(
    title,
    description
):

    text = (
        title
        + " "
        + description
    )


    clean_text = processor.clean_text(
        text
    )


    category = category_model.predict(
        [clean_text]
    )[0]


    priority = priority_model.predict(
        [clean_text]
    )[0]


    result = {

        "category": category,

        "priority": priority,

        "assigned_team":
            TEAM_MAPPING.get(
                category,
                "IT Helpdesk"
            ),

        "suggested_solution":
            SOLUTIONS.get(
                category,
                "Contact IT Helpdesk."
            ),

        "summary":
            f"{title}: "
            f"{description[:200]}"
    }


    # Category confidence

    category_probability = (

        category_model
        .predict_proba(
            [clean_text]
        )[0]
    )


    result[
        "category_confidence"
    ] = round(

        float(
            max(
                category_probability
            )
        ) * 100,

        2
    )


    # Priority confidence

    priority_probability = (

        priority_model
        .predict_proba(
            [clean_text]
        )[0]
    )


    result[
        "priority_confidence"
    ] = round(

        float(
            max(
                priority_probability
            )
        ) * 100,

        2
    )


    return result


# ---------------------------------------
# Similar tickets
# ---------------------------------------

def similar_tickets(
    text,
    limit=5
):

    clean_text = processor.clean_text(
        text
    )


    query_vector = (

        similarity_vectorizer
        .transform(
            [clean_text]
        )
    )


    scores = cosine_similarity(

        query_vector,

        similarity_matrix
    )[0]


    indices = scores.argsort()[::-1]


    results = []


    for index in indices[:limit]:

        results.append({

            "title":
                df.iloc[index]["title"],

            "description":
                df.iloc[index]["description"],

            "category":
                df.iloc[index]["category"],

            "priority":
                df.iloc[index]["priority"],

            "similarity":
                round(
                    float(
                        scores[index]
                    ) * 100,

                    2
                )
        })


    return results