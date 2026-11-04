
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. Load cleaned data
df = pd.read_csv("chatgpt_cleaned_reviews.csv")

df["Cleaned_Text"] = df["Cleaned_Text"].fillna("").astype(str)
df = df[df["Cleaned_Text"].str.strip() != ""].copy()

# Use a reproducible sample to keep plotting fast
if len(df) > 4000:
    df = df.sample(n=4000, random_state=42)

print("Reviews used:", len(df))

# 2. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=2,
    ngram_range=(1, 2)
)

tfidf = vectorizer.fit_transform(df["Cleaned_Text"])

print("TF-IDF shape:", tfidf.shape)

# 3. Reduce the features to two dimensions
svd = TruncatedSVD(n_components=2, random_state=42)
reduced_data = svd.fit_transform(tfidf)

print(
    "Variance explained by two components:",
    f"{svd.explained_variance_ratio_.sum() * 100:.2f}%"
)

# 4. Plot the reviews, grouped by rating-derived sentiment
plt.figure(figsize=(10, 7))

colors = {
    "Positive": "green",
    "Neutral": "orange",
    "Negative": "red"
}

for sentiment in ["Positive", "Neutral", "Negative"]:
    mask = df["Sentiment"].to_numpy() == sentiment

    plt.scatter(
        reduced_data[mask, 0],
        reduced_data[mask, 1],
        label=sentiment,
        color=colors[sentiment],
        alpha=0.55,
        s=18
    )

plt.title("Dimensionality Reduction of ChatGPT Reviews (Truncated SVD)")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.legend(title="Rating-derived sentiment")
plt.tight_layout()

plt.savefig(
    "dimensionality_reduction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: dimensionality_reduction.png")
