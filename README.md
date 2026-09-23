# AI-Powered IT Support Ticket Classification and Auto-Routing System

## Overview

The AI-Powered IT Support Ticket Classification and Auto-Routing System is an NLP-based application that automatically analyzes IT support tickets and predicts the ticket category, priority, assigned support team, prediction confidence, and suggested solution.

The system combines Natural Language Processing, Machine Learning, FastAPI, PostgreSQL, and a web frontend to create an end-to-end IT support ticket management application.

## Features

NLP-based ticket classification

Automatic ticket category prediction

Automatic priority prediction

Automatic support-team assignment

Category and priority confidence estimates

Similar historical ticket retrieval using cosine similarity

Category-based suggested solutions

PostgreSQL storage for submitted tickets

Web interface for submitting and viewing tickets

FastAPI REST API

Model evaluation using 5-fold cross-validation

Comparison of Logistic Regression and Linear SVM during evaluation

## System Architecture

User

↓

Web Frontend

↓

FastAPI

↓

NLP Preprocessing

↓

TF-IDF

↓

Category Model and Priority Model

↓

Prediction Result

↓

Category, Priority, Assigned Team, Suggested Solution

↓

PostgreSQL

## Machine Learning Approach

### NLP Preprocessing

The ticket title and description are combined and processed using NLTK.

The preprocessing pipeline includes lowercase conversion, special character removal, tokenization, stop-word removal, removal of very short tokens, and lemmatization.

Example:

Original:

VPN is NOT connecting to the company network.

After preprocessing:

vpn connecting company network

### TF-IDF

The cleaned ticket text is converted into numerical features using TF-IDF.

The project uses unigrams and bigrams with sublinear term frequency and a maximum of 10,000 features.

### Classification

Two separate classification pipelines are trained.

TF-IDF followed by Logistic Regression is used for category prediction.

TF-IDF followed by Logistic Regression is used for priority prediction.

The category model predicts Network, Hardware, Software, Database, Access, Security, Application, Email, and Cloud.

The priority model predicts Low, Medium, High, and Critical.

## Model Evaluation

The project evaluates different machine-learning algorithms using stratified 5-fold cross-validation.

The evaluated models include Logistic Regression and Linear SVM.

The evaluation measures accuracy, precision, recall, and F1-score.

The dataset is divided into five folds. Each fold is used for validation once while the remaining four folds are used for training.

Linear SVM is used as an evaluation candidate. The current application uses Logistic Regression because the application also displays probability-based confidence estimates using predict_proba.

## Similar Ticket Search

The system provides similar historical tickets using cosine similarity.

The process is:

New ticket

↓

NLP preprocessing

↓

TF-IDF

↓

Cosine similarity

↓

Historical tickets ranked by similarity

This allows the application to find previously recorded tickets that are similar to the current issue.

## PostgreSQL Integration

PostgreSQL is used for persistent runtime ticket storage.

When a user submits a ticket, the system stores the ticket ID, title, description, predicted category, predicted priority, assigned team, category confidence, priority confidence, suggested solution, and creation timestamp.

The current training dataset remains separate in data/tickets.csv.

The system does not automatically retrain using its own predictions because predictions may be incorrect.

## Project Structure

it-support-nlp/

app/

**init**.py

main.py

nlp.py

services.py

database.py

backend/

download_nltk.py

nlp.py

test_nlp.py

data/

tickets.csv

models/

category_model.pkl

priority_model.pkl

preprocessor.pkl

evaluation/

static/

index.html

train.py

requirements.txt

Dockerfile

.gitignore

.python-version

.env

## Technologies Used

Python

NLTK

Pandas

Scikit-learn

TF-IDF

Logistic Regression

Linear SVM

Cosine Similarity

FastAPI

PostgreSQL

HTML

CSS

JavaScript

Uvicorn

## Installation

### Clone the Repository

git clone [https://github.com/harini-012/it-support-nlp.git](https://github.com/harini-012/it-support-nlp.git)

cd it-support-nlp

### Create a Virtual Environment

python -m venv venv

Activate the environment on Windows:

venv\Scripts\activate

### Install Dependencies

pip install -r requirements.txt

### Configure PostgreSQL

Create a PostgreSQL database named it_support_db.

Create a .env file in the project root.

DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/it_support_db

Replace YOUR_PASSWORD with your PostgreSQL password.

### Download NLTK Resources

python backend/download_nltk.py

## Train the Models

Make sure the training data exists at:

data/tickets.csv

Run:

python train.py

The training process creates:

models/category_model.pkl

models/priority_model.pkl

models/preprocessor.pkl

## Run the Application

Start the FastAPI server:

uvicorn app.main:app --reload

Open the application in a browser at:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

## API Endpoints

### Health Check

GET /api/health

### Predict Ticket

POST /api/predict

Example request:

{
"title": "VPN is not connecting",
"description": "I cannot connect to the company VPN from my laptop."
}

### Retrieve Ticket History

GET /api/tickets

Returns previously submitted tickets stored in PostgreSQL.

### Find Similar Tickets

POST /api/similar

Example request:

{
"text": "VPN connection is failing"
}

Returns similar historical tickets based on cosine similarity.

## Example Workflow

A user submits a ticket.

Title:

VPN is not working

Description:

I cannot connect to the company VPN.

The system processes the ticket through NLP preprocessing, TF-IDF, category prediction, priority prediction, team assignment, suggested solution generation, and PostgreSQL storage.

Example result:

Category: Network

Priority: High

Assigned Team: Network Support

Category Confidence: 92 percent

Priority Confidence: 87 percent

## Training and Runtime

Training uses the following process:

tickets.csv

↓

Preprocessing

↓

TF-IDF

↓

Logistic Regression

↓

Saved Model

Runtime prediction uses the following process:

User Ticket

↓

FastAPI

↓

Preprocessing

↓

Saved Model

↓

Prediction

↓

PostgreSQL

The model is not retrained every time a user submits a ticket.

## Future Enhancements

Larger and more diverse training dataset

Human validation of predictions

Periodic retraining using verified historical tickets

Transformer-based NLP models

Authentication and role-based access

Admin dashboard

Ticket status management

Human correction of incorrect predictions

Advanced semantic similarity using embeddings

Model monitoring and performance tracking

## Important Design Decision

The system does not automatically use its own predictions as new training labels.

For example, if the model predicts Network, the prediction is not automatically treated as the actual category.

A prediction can be incorrect. Therefore, future retraining should use verified or human-corrected labels rather than automatically training on the model's own predictions.

## Author

Harini



## Project Summary

An NLP-powered IT support ticket classification and auto-routing system that uses TF-IDF and Logistic Regression to predict ticket categories and priorities, automatically assigns support teams, retrieves similar historical tickets using cosine similarity, exposes predictions through FastAPI, and stores runtime ticket history in PostgreSQL.
