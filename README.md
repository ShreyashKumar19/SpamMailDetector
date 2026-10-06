# Spam Mail Detector

This is my second machine learning project.

## Objective

The aim of this project is to classify messages as Spam or Ham (not spam).

## Dataset

I used the SMS Spam Collection dataset from the UCI Machine Learning Repository.

The dataset contains 5,574 SMS messages. Each message has a label:
* ham = normal message
* spam = spam message

UCI dataset:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Files

* 'spam_mail_detector.py' - main project code
* 'requirements.txt' - required libraries
* 'download_dataset.py' - downloads the UCI dataset
* 'SMSSpamCollection' - dataset file after downloading

## How to Run

First install the required libraries:

pip install -r requirements.txt

Then download the dataset:

python download_dataset.py

After that, run the main program:

python spam_mail_detector.py

## Steps Used

1. Load the messages and labels
2. Convert text to lowercase
3. Remove special characters
4. Remove common English stopwords
5. Convert text into numerical features using TF-IDF
6. Split the data into training and testing data
7. Train a Naive Bayes model
8. Check accuracy, precision, recall and F1-score
9. Display a confusion matrix
10. Test a new message

## Model

I used Multinomial Naive Bayes for classification because it is a simple model that works well with text data.