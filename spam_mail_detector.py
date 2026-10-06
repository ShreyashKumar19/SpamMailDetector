import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load the dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nNumber of messages:")
print(df["label"].value_counts())


# Show the number of spam and ham messages
sns.countplot(x="label", data=df)

plt.title("Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")

plt.show()


# Convert spam and ham into numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# Convert messages to lowercase
df["message"] = df["message"].str.lower()


# Remove special characters
df["message"] = df["message"].str.replace(
    r"[^a-zA-Z0-9\s]",
    "",
    regex=True
)


# Separate messages and labels
X = df["message"]
y = df["label"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Create and train the model
model = MultinomialNB()

model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Check accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))


# Show precision, recall and F1-score
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# Test a new message
message = [
    "Congratulations! You have won a free prize. Click here to claim it."
]

message = vectorizer.transform(message)

prediction = model.predict(message)

if prediction[0] == 1:
    print("\nNew Message: Spam")
else:
    print("\nNew Message: Ham")
