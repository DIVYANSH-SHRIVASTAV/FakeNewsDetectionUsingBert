# A Novel Approach for Identifying Fake News through Optimized BERT

This project demonstrates how to use BERT (Bidirectional Encoder Representations from Transformers) to build a fake news detection model.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)
- [Prediction Function](#prediction-function)
- [Web Interface](#web-interface)


## Introduction

Fake news is a growing problem with serious consequences. This project aims to develop a model that can automatically detect fake news articles. We leverage the power of BERT, a state-of-the-art language model, to achieve this goal.

## Dataset

The project uses the "[ISOT Fake News dataset](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset) 🔗
" dataset which has two files for fake and real news in csv format containing title,text, subject and date content of the articles.
The dataset is processed and cleaned within this project.

## Installation

1. To run this project, clone the ripository [(  https://github.com/DIVYANSH-SHRIVASTAV/FakeNewsDetectionUsingBert  )]🔗: 

2. Install following libraries : [pip install -r requirements.txt]()
    - transformers 
    - pandas 
    - scikit-learn 
    - seaborn 
    - matplotlib 
    - wordcloud 
    - gensim 
    - nltk 
    - tensorflow

3. Run the training script: python FND_Code.ipynb

4. Run Web App: python app.py

## Requirements
- Python 3.8+
- Pytorch
- Transformer (Hugging Face)
- Gradio
## Usage

1. **Mount Google Drive:** Mount your Google Drive to access the dataset stored there.
2. **Load Data:** Load the true and fake news datasets using pandas.
3. **Data Preprocessing:** Clean and preprocess the data (removing stop words, stemming, tokenization, padding).
4. **Model Construction:** Load the pre-trained BERT model for sequence classification.
5. **Fine-tuning:** Fine-tune the BERT model on the training data.
6. **Model Saving and Loading:** Save and load the trained model for later use.
7. **Testing and Evaluation:** Test the model on the test data and evaluate its performance.

## Model

The project uses a pre-trained BERT model for sequence classification. The model is fine-tuned on the ISOT fake news dataset to achieve high accuracy in detecting fake news articles.

## Evaluation

The model is evaluated using various metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Prediction Function

A function `predict_fake_news(text)` is provided to predict whether a given news article is fake or not.

## Author
- Name : Divyansh Shrivastav
- Email : srivastavdivyansh9@gmail.com



#   F a k e N e w s D e t e c t i o n U s i n g B e r t 
 
 
