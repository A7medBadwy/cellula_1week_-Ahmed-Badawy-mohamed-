# Toxic Comment Classification using RNN & LSTM

A deep learning project for detecting toxic comments using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks with PyTorch.

The project is based on the Jigsaw Toxic Comment Classification dataset and performs multi-label text classification.

## Project Overview

The goal of this project is to classify online comments into multiple toxicity categories.

Each comment can belong to more than one category.

The target classes are:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

This makes the task a multi-label classification problem.

## Dataset

The project uses the Jigsaw Toxic Comment Classification dataset.

The training dataset contains:

- `comment_text`
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The test dataset contains the comments that are used for final predictions.

## Technologies

- Python
- PyTorch
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

## NLP Pipeline

The text preprocessing pipeline includes:

1. Convert text to lowercase
2. Remove HTML tags
3. Remove URLs
4. Remove non-alphabetic characters
5. Normalize whitespace
6. Tokenization using NLTK
7. Lemmatization
8. Vocabulary creation
9. Integer encoding
10. Padding sequences

Stop words were kept during preprocessing.

## Model Architecture

The project experiments with Recurrent Neural Network architectures for text classification.

### RNN

The RNN architecture contains:

- Embedding layer
- Bidirectional RNN
- Dropout
- Fully Connected layers
- Multi-label output layer

### LSTM

The LSTM model is designed to capture long-term dependencies in text.

The architecture includes:

- Embedding layer
- Bidirectional LSTM
- Dropout
- Fully Connected layers
- Sigmoid output

The model produces one probability for each target class.

## Multi-Label Classification

Since a comment can belong to multiple categories, the model uses a sigmoid activation for the output layer.

