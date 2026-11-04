
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# 1. Load cleaned reviews
df = pd.read_csv("chatgpt_cleaned_reviews.csv")

df["Cleaned_Text"] = df["Cleaned_Text"].fillna("").astype(str)
df["Sentiment"] = df["Sentiment"].fillna("").astype(str)

df = df[
    (df["Cleaned_Text"].str.strip() != "") &
    (df["Sentiment"].isin(["Positive", "Neutral", "Negative"]))
].copy()

print("Dataset shape:", df.shape)
print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

# 2. Separate input text and target labels
X = df["Cleaned_Text"]
y = df["Sentiment"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# 4. Convert text into TF-IDF features
# Fit only on training data to prevent test-data leakage.
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

# 5. Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

print("\nTraining the model...")
model.fit(X_train_tfidf, y_train)

# 6. Predict sentiment on unseen test reviews
y_pred = model.predict(X_test_tfidf)

# 7. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1-score: {macro_f1:.4f}")

report = classification_report(
    y_test,
    y_pred,
    labels=["Negative", "Neutral", "Positive"],
    target_names=["Negative", "Neutral", "Positive"],
    zero_division=0,
    output_dict=True
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["Negative", "Neutral", "Positive"],
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    )
)

# Save classification metrics
pd.DataFrame(report).transpose().to_csv(
    "classification_report.csv"
)

# 8. Create and save confusion matrix
labels = ["Negative", "Neutral", "Positive"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Rating-Based Sentiment")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# 9. Save actual and predicted results for analysis
results = pd.DataFrame({
    "Review": X_test.values,
    "Actual_Sentiment": y_test.values,
    "Predicted_Sentiment": y_pred
})

results.to_csv("sentiment_predictions.csv", index=False)

# 10. Save model and vectorizer for later use in the dashboard
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# 11. Try predicting a new review
sample_review = [
    "This app is very helpful and gives good answers"
]

sample_features = vectorizer.transform(sample_review)
sample_prediction = model.predict(sample_features)[0]

print("\n========== SAMPLE PREDICTION ==========")
print("Review:", sample_review[0])
print("Predicted sentiment:", sample_prediction)

print("\nSaved files:")
print("- classification_report.csv")
print("- confusion_matrix.png")
print("- sentiment_predictions.csv")
print("- sentiment_model.pkl")
print("- tfidf_vectorizer.pkl")
