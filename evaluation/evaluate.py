import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.naive_bayes import (
    MultinomialNB
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.svm import (
    LinearSVC
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATA_PATH = os.path.join(
    ROOT,
    "data",
    "tickets.csv"
)


OUTPUT_DIR = os.path.join(
    ROOT,
    "evaluation"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------
# Load data
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
# Models
# ---------------------------------------

models = {

    "Naive Bayes": Pipeline([

        (
            "tfidf",

            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),

        (
            "classifier",

            MultinomialNB()
        )
    ]),


    "Logistic Regression": Pipeline([

        (
            "tfidf",

            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=3000
            )
        )
    ]),


    "Linear SVM": Pipeline([

        (
            "tfidf",

            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),

        (
            "classifier",

            LinearSVC()
        )
    ])
}


# ---------------------------------------
# Holdout evaluation
# ---------------------------------------

all_results = []


for target in [
    "category",
    "priority"
]:

    print("\n")
    print("=" * 70)

    print(
        f"{target.upper()} CLASSIFICATION"
    )

    print("=" * 70)


    X_train, X_test, y_train, y_test = train_test_split(

        df["text"],

        df[target],

        test_size=0.25,

        random_state=42,

        stratify=df[target]
    )


    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            predictions
        )


        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        all_results.append({

            "Task": target,

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1": f1
        })


        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )


        # Classification report

        report = classification_report(

            y_test,

            predictions,

            zero_division=0
        )


        report_path = os.path.join(

            OUTPUT_DIR,

            f"{target}_{name.replace(' ', '_')}_report.txt"
        )


        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(report)


        print("\nClassification Report:")
        print(report)


        # Confusion matrix

        labels = sorted(
            df[target].unique()
        )


        matrix = confusion_matrix(

            y_test,

            predictions,

            labels=labels
        )


        plt.figure(
            figsize=(11, 8)
        )


        sns.heatmap(

            matrix,

            annot=True,

            fmt="d",

            xticklabels=labels,

            yticklabels=labels
        )


        plt.xlabel(
            "Predicted"
        )


        plt.ylabel(
            "Actual"
        )


        plt.title(
            f"{target.title()} - {name}"
        )


        plt.tight_layout()


        image_path = os.path.join(

            OUTPUT_DIR,

            f"{target}_{name.replace(' ', '_')}_confusion_matrix.png"
        )


        plt.savefig(
            image_path
        )


        plt.close()


# ---------------------------------------
# Model comparison
# ---------------------------------------

results = pd.DataFrame(
    all_results
)


results_path = os.path.join(

    OUTPUT_DIR,

    "model_comparison.csv"
)


results.to_csv(
    results_path,
    index=False
)


print("\n")
print("=" * 70)

print(
    "MODEL COMPARISON"
)

print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


# ---------------------------------------
# 5-fold cross validation
# ---------------------------------------

print("\n")
print("=" * 70)

print(
    "5-FOLD CROSS VALIDATION"
)

print("=" * 70)


cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


scoring = [

    "accuracy",

    "precision_weighted",

    "recall_weighted",

    "f1_weighted"
]


for target in [
    "category",
    "priority"
]:

    print(
        f"\nTARGET: {target}"
    )


    for name, model in models.items():

        scores = cross_validate(

            model,

            df["text"],

            df[target],

            cv=cv,

            scoring=scoring
        )


        print(
            f"\n{name}"
        )


        print(
            "CV Accuracy :",
            round(
                scores[
                    "test_accuracy"
                ].mean(),
                4
            )
        )


        print(
            "CV Precision:",
            round(
                scores[
                    "test_precision_weighted"
                ].mean(),
                4
            )
        )


        print(
            "CV Recall   :",
            round(
                scores[
                    "test_recall_weighted"
                ].mean(),
                4
            )
        )


        print(
            "CV F1       :",
            round(
                scores[
                    "test_f1_weighted"
                ].mean(),
                4
            )
        )