
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load the cleaned dataset
df = pd.read_csv("chatgpt_cleaned_reviews.csv")

df["Cleaned_Text"] = df["Cleaned_Text"].fillna("").astype(str)
df = df[df["Cleaned_Text"].str.strip() != ""].copy()

print("Dataset loaded:", df.shape)

# 2. Extract TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2
)

tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Text"])
terms = vectorizer.get_feature_names_out()

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# 3. Save TF-IDF matrix efficiently
from scipy.sparse import save_npz

save_npz("chatgpt_tfidf_matrix.npz", tfidf_matrix)

pd.DataFrame({"Term": terms}).to_csv(
    "tfidf_feature_names.csv", index=False
)

# 4. Find the top terms by average TF-IDF score
average_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

top_indices = np.argsort(average_scores)[::-1][:20]

top_tfidf = pd.DataFrame({
    "Term": terms[top_indices],
    "Average_TFIDF": average_scores[top_indices]
})

top_tfidf.to_csv("tfidf_top_terms.csv", index=False)

print("\nTop 20 TF-IDF terms:")
print(top_tfidf.to_string(index=False))

# 5. Generate a word cloud
all_text = " ".join(df["Cleaned_Text"])

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=150
).generate(all_text)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of ChatGPT User Reviews")
plt.tight_layout()
plt.savefig("wordcloud.png", dpi=300, bbox_inches="tight")
plt.show()

# 6. Plot the top 20 frequent words
from collections import Counter

word_counts = Counter(all_text.split())
top_words = word_counts.most_common(20)

words_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
words_df = words_df.sort_values("Frequency")

plt.figure(figsize=(10, 8))
plt.barh(words_df["Word"], words_df["Frequency"])
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.title("Top 20 Most Frequent Words in ChatGPT Reviews")
plt.tight_layout()
plt.savefig("top_word_frequency.png", dpi=300, bbox_inches="tight")
plt.show()

# 7. Plot sentiment distribution
sentiment_counts = df["Sentiment"].value_counts().reindex(
    ["Positive", "Neutral", "Negative"], fill_value=0
)

plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Sentiment Distribution of ChatGPT Reviews")
plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# 8. Create a heat map of common words by sentiment
# Values represent the percentage of reviews in each class
# that contain each word.
top_15_words = [word for word, count in word_counts.most_common(15)]

heatmap_data = pd.DataFrame(index=top_15_words)

for sentiment in ["Positive", "Neutral", "Negative"]:
    subset = df.loc[df["Sentiment"] == sentiment, "Cleaned_Text"]
    heatmap_data[sentiment] = [
        subset.str.contains(
            rf"\b{word}\b", regex=True, na=False
        ).mean() * 100
        for word in top_15_words
    ]

plt.figure(figsize=(10, 9))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)
plt.title("Percentage of Reviews Containing Common Words by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Word")
plt.tight_layout()
plt.savefig("sentiment_word_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# 9. Report the output files
print("\nAll outputs saved in:")
print(Path.cwd())

print("\nGenerated files:")
for filename in [
    "chatgpt_tfidf_matrix.npz",
    "tfidf_feature_names.csv",
    "tfidf_top_terms.csv",
    "wordcloud.png",
    "top_word_frequency.png",
    "sentiment_distribution.png",
    "sentiment_word_heatmap.png"
]:
    print("-", filename)
