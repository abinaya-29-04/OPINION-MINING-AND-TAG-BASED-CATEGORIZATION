
import re
import unicodedata
from pathlib import Path

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK resources (only needed the first time)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

DATA_FILE = "The_ Generative_AI_Ecosystem_50k_User_Reviews_2026.csv"

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

print("Original dataset:", df.shape)

# 2. Select ChatGPT reviews
df = df[df["App"] == "ChatGPT"].copy()

# 3. Remove missing and duplicate reviews
df["Review_Text"] = df["Review_Text"].fillna("").astype(str).str.strip()
df = df[df["Review_Text"] != ""]
df = df.drop_duplicates(subset=["Review_Text"])

# 4. Convert stylized Unicode characters to normal text
def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Normalized_Text"] = df["Review_Text"].apply(normalize_text)

# 5. Tokenize, remove stop words, and lemmatize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    tokens = text.split()
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(cleaned_tokens)

df["Cleaned_Text"] = df["Normalized_Text"].apply(clean_text)

# Remove reviews that become empty after cleaning
df = df[df["Cleaned_Text"].str.strip() != ""].copy()

# 6. Create three sentiment classes from ratings
def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Star_Rating"].apply(rating_to_sentiment)

# 7. Save the cleaned data
output_file = "chatgpt_cleaned_reviews.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

# 8. Create Bag-of-Words matrix
vectorizer = CountVectorizer(max_features=1000)
bow_matrix = vectorizer.fit_transform(df["Cleaned_Text"])

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)
bow_df.to_csv("chatgpt_bow_matrix.csv", index=False)

# 9. Display results
print("\nCleaned dataset shape:", df.shape)
print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

print("\nTop 20 frequent words:")
print(
    bow_df.sum()
    .sort_values(ascending=False)
    .head(20)
)

print("\nSaved files:")
print(Path(output_file).resolve())
print(Path("chatgpt_bow_matrix.csv").resolve())
