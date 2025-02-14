# Disaster Tweet Classification

## Overview
This repository contains my submission for a machine learning competition aimed at classifying tweets as either related to real disasters or not. The goal of this competition was to build a model capable of distinguishing between tweets that announce real emergencies and those that do not.

## Problem Statement
Twitter has become an essential communication platform during emergencies, allowing people to share real-time updates on disasters. However, not every tweet mentioning a disaster-related term is actually about an emergency. Disaster relief organizations and news agencies require an automated way to filter through tweets and identify real disaster-related messages.

This project focuses on solving this problem by leveraging machine learning techniques to accurately classify tweets into disaster-related and non-disaster-related categories.

## Dataset
The dataset consists of 10,000 tweets that have been manually labeled as either referring to real disasters (1) or not (0). The dataset includes:
- `id`: Unique identifier for each tweet
- `text`: The tweet content
- `location`: (Optional) Location mentioned in the tweet
- `keyword`: (Optional) Disaster-related keyword
- `target`: The label (1 = Disaster, 0 = Not a disaster)

## Approach
### 1. Data Preprocessing
- Removed URLs, hashtags, and special characters.
- Tokenization and lemmatization of words.
- Used stopword removal to eliminate irrelevant words.
- Employed TF-IDF for text vectorization.

### 2. Model Selection
- **Baseline Model:** Bernoulli Naive Bayes

### 3. Evaluation
The models were evaluated using:
- Accuracy
- Precision, Recall, and F1-score
- ROC-AUC score

## Results
The model achieved an F1-score of **0.79313** on the test set.

## How to Use
### 1. Clone the Repository
```sh
git clone https://github.com/udensidev/disaster-tweet-classification.git
cd disaster-tweet-classification
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the Model
To train the model and generate predictions:
```sh
python train.py
```
To evaluate the model:
```sh
python evaluate.py
```

## Future Work
- Employing transformer-based models for better performance.
- Expanding the dataset with more diverse tweets.