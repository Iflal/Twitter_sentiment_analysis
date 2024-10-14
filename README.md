# Twitter Sentiment Analysis

This project performs sentiment analysis on tweets using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), which contains 1.6 million tweets labeled as positive or negative. The main objective is to train a machine learning model to predict whether a given tweet expresses positive or negative sentiment. 

**Note**: This project was created as part of my learning journey in natural language processing and machine learning.

## Project Overview

Sentiment analysis is a Natural Language Processing (NLP) technique used to identify and extract subjective information from text. This project aims to classify the sentiment of tweets as positive or negative, which can help businesses, researchers, or social media managers analyze public opinion on various topics or products.

### Features

- **Dataset**: 1.6 million tweets from the Sentiment140 dataset.
- **Preprocessing**: Data cleaning, tokenization, and text vectorization techniques applied.
- **Model Training**: A machine learning model was trained to classify tweet sentiment.
- **Prediction**: The model predicts the sentiment of unseen tweets as either positive or negative.
- **Evaluation**: Model performance was evaluated using accuracy.

## Dataset

The **Sentiment140 dataset** includes the following fields:

- **target**: Sentiment of the tweet (0 = negative, 4 = positive)
- **id**: Unique ID of the tweet
- **date**: Date when the tweet was created
- **user**: Username of the person who tweeted
- **text**: The content of the tweet

## Project Workflow

1. **Data Collection**:
   - The dataset was sourced from [Kaggle's Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).

2. **Data Preprocessing**:
   - Removed unwanted characters (e.g., URLs, mentions, hashtags).
   - Tokenized the text (splitting it into individual words).
   - Removed stopwords (common words like "the", "is", etc.).
   - Applied techniques like stemming/lemmatization to reduce words to their base forms.
   - Vectorized the text using TF-IDF or Word Embeddings (depending on the model).

   **Important Note**: The stemming process is time-consuming. In Google Colab, it takes around 50 minutes, while locally it can take up to 150 minutes.

3. **Model Training**:
   - Trained a logistic regression model to classify tweet sentiment.

4. **Model Evaluation**:
   - Split the dataset into training and testing sets (80% training, 20% testing).
   - Evaluated model performance using accuracy.

5. **Prediction**:
   - The best-performing model was used to predict the sentiment of new, unseen tweets.

## Results

The model achieved the following performance on the test set:

- **Accuracy**: 80%

The model demonstrates a good ability to classify tweet sentiment based on the content.

## Installation and Usage

### Requirements

- Python 3.x
- Jupyter Notebook (optional, for running the project interactively)
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - nltk (for text preprocessing)

### Steps to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/iflal/twitter_sentiment_analysis.git
cd twitter_sentiment_analysis
