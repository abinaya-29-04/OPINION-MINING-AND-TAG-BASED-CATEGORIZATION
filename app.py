
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="ChatGPT Opinion Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


def file_path(filename):
    return BASE_DIR / filename


# ==================================================
# 2. LOAD DATA AND TRAINED MODEL
# ==================================================
@st.cache_data
def load_reviews():
    return pd.read_csv(file_path("chatgpt_cleaned_reviews.csv"))


@st.cache_data
def load_report():
    path = file_path("classification_report.csv")

    if path.exists():
        return pd.read_csv(path, index_col=0)

    return None


@st.cache_resource
def load_model():
    model = joblib.load(file_path("sentiment_model.pkl"))
    vectorizer = joblib.load(file_path("tfidf_vectorizer.pkl"))

    return model, vectorizer


df = load_reviews()
report = load_report()
model, vectorizer = load_model()


# ==================================================
# 3. PREPROCESS NEW REVIEWS
# ==================================================
@st.cache_resource
def get_text_tools():
    try:
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        # Check whether WordNet is available.
        lemmatizer.lemmatize("reviews")

        return stop_words, lemmatizer

    except LookupError:
        return None, None


def clean_new_review(text):
    stop_words, lemmatizer = get_text_tools()

    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    if stop_words is None:
        return " ".join(
            word for word in tokens if len(word) > 1
        )

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(cleaned_tokens)


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("📊 Opinion Mining")
st.sidebar.caption("ChatGPT User Review Analysis")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Text Analysis",
        "Sentiment Model",
        "Predict a Review",
        "Dataset Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.write("Dataset: ChatGPT user reviews")
st.sidebar.write(f"Cleaned reviews: {len(df):,}")


# ==================================================
# 5. COMMON HEADER AND SENTIMENT DATA
# ==================================================
st.title("📊 Opinion Mining and Tag-Based Categorization")
st.subheader("Analysis of ChatGPT User Feedback")

st.write(
    "Explore review sentiment, important terms, TF-IDF features, "
    "model evaluation, dimensionality reduction, and predictions "
    "for new reviews."
)

sentiment_order = ["Positive", "Neutral", "Negative"]

sentiment_counts = (
    df["Sentiment"]
    .value_counts()
    .reindex(sentiment_order, fill_value=0)
)

total_reviews = len(df)

positive_count = int(sentiment_counts["Positive"])
neutral_count = int(sentiment_counts["Neutral"])
negative_count = int(sentiment_counts["Negative"])


# ==================================================
# 6. OVERVIEW PAGE
# ==================================================
if page == "Overview":

    st.markdown("## Review Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", f"{total_reviews:,}")

    col2.metric(
        "Positive Reviews",
        f"{positive_count:,}",
        f"{positive_count / total_reviews:.1%}"
        if total_reviews else "0%"
    )

    col3.metric(
        "Negative Reviews",
        f"{negative_count:,}",
        f"{negative_count / total_reviews:.1%}"
        if total_reviews else "0%"
    )

    col4.metric(
        "Neutral Reviews",
        f"{neutral_count:,}",
        f"{neutral_count / total_reviews:.1%}"
        if total_reviews else "0%"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Sentiment Distribution")

        sentiment_chart = (
            sentiment_counts
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )

        st.bar_chart(
            sentiment_chart,
            x="Sentiment",
            y="Reviews"
        )

    with right:
        st.markdown("### Sentiment Proportions")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Preview of Cleaned Reviews")

    preview_columns = [
        col for col in
        ["Review_Date", "Star_Rating", "Review_Text", "Sentiment"]
        if col in df.columns
    ]

    st.dataframe(
        df[preview_columns].head(10),
        use_container_width=True
    )

    st.info(
        "Sentiment labels are derived from star ratings: "
        "4–5 stars = Positive, 3 stars = Neutral, "
        "and 1–2 stars = Negative."
    )


# ==================================================
# 7. TEXT ANALYSIS PAGE
# ==================================================
elif page == "Text Analysis":

    st.markdown("## Word Cloud")

    wordcloud_path = file_path("wordcloud.png")

    if wordcloud_path.exists():
        st.image(
            str(wordcloud_path),
            use_container_width=True
        )
    else:
        st.warning(
            "Run tfidf_visualization.py to create the word cloud."
        )

    # Word frequency and heat map
    left, right = st.columns(2)

    with left:
        st.markdown("### Top Word Frequencies")

        path = file_path("top_word_frequency.png")

        if path.exists():
            st.image(
                str(path),
                use_container_width=True
            )
        else:
            st.warning("Word frequency chart not found.")

    with right:
        st.markdown("### Word Frequency by Sentiment")

        path = file_path("sentiment_word_heatmap.png")

        if path.exists():
            st.image(
                str(path),
                use_container_width=True
            )
        else:
            st.warning("Sentiment heat map not found.")

    st.divider()

    # TF-IDF analysis
    st.markdown("### TF-IDF Analysis")

    tfidf_path = file_path("tfidf_top_terms.csv")

    if tfidf_path.exists():

        tfidf_df = pd.read_csv(tfidf_path)

        st.dataframe(
            tfidf_df,
            use_container_width=True
        )

        st.markdown("#### Top TF-IDF Terms")

        st.bar_chart(
            tfidf_df.head(20).set_index("Term")["Average_TFIDF"]
        )

    else:
        st.warning(
            "Run tfidf_visualization.py to generate TF-IDF results."
        )

    st.caption(
        "TF-IDF measures the relative importance of terms in the "
        "analyzed corpus. It is not a sentiment score."
    )

    st.divider()

    # NEW: DIMENSIONALITY REDUCTION PLOT
    st.markdown("### Dimensionality Reduction (SVD)")

    svd_path = file_path("dimensionality_reduction.png")

    if svd_path.exists():
        st.image(
            str(svd_path),
            use_container_width=True
        )
        st.caption(
            "Each point represents a review projected into two "
            "dimensions using Truncated SVD. Colors represent "
            "rating-derived sentiment. Overlap is expected, and "
            "the plot does not show all information in the original "
            "TF-IDF feature space."
        )
    else:
        st.warning(
            "Run dimensionality_reduction.py first."
        )


# ==================================================
# 8. SENTIMENT MODEL PAGE
# ==================================================
elif page == "Sentiment Model":

    st.markdown("## Logistic Regression Performance")

    cm_path = file_path("confusion_matrix.png")

    if cm_path.exists():
        st.markdown("### Confusion Matrix")

        st.image(
            str(cm_path),
            use_container_width=False
        )
    else:
        st.warning(
            "Run sentiment_classifier.py to create the confusion matrix."
        )

    if report is not None:

        st.markdown("### Classification Report")

        st.dataframe(
            report,
            use_container_width=True
        )

        if (
            "macro avg" in report.index
            and "f1-score" in report.columns
        ):
            macro_f1 = report.loc["macro avg", "f1-score"]

            st.metric(
                "Macro F1-score",
                f"{macro_f1:.4f}"
            )

        if (
            "accuracy" in report.index
            and "f1-score" in report.columns
        ):
            accuracy = report.loc["accuracy", "f1-score"]

            st.metric(
                "Test Accuracy",
                f"{accuracy:.2%}"
            )

    else:
        st.warning("Classification report CSV was not found.")

    prediction_path = file_path("sentiment_predictions.csv")

    if prediction_path.exists():

        results = pd.read_csv(prediction_path)

        st.markdown("### Actual vs Predicted Sentiment")

        st.dataframe(
            results.head(100),
            use_container_width=True
        )

    st.warning(
        "Test labels are derived from star ratings rather than "
        "independent human sentiment annotations. The Neutral class "
        "has a lower F1-score than the other classes."
    )


# ==================================================
# 9. PREDICT A NEW REVIEW PAGE
# ==================================================
elif page == "Predict a Review":

    st.markdown("## Try Sentiment Prediction")

    st.write(
        "Enter a review to estimate its rating-derived sentiment."
    )

    review_text = st.text_area(
        "Enter review text",
        placeholder=(
            "Example: This app is very helpful and gives good answers"
        ),
        height=140
    )

    if st.button("Analyze Sentiment", type="primary"):

        if not review_text.strip():

            st.warning("Please enter a review first.")

        else:

            cleaned = clean_new_review(review_text)

            if not cleaned:

                st.warning(
                    "The review became empty after preprocessing. "
                    "Try entering a longer review."
                )

            else:

                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)[0]

                st.markdown("### Prediction")

                if prediction == "Positive":
                    st.success(
                        f"Predicted sentiment: {prediction} 😊"
                    )

                elif prediction == "Negative":
                    st.error(
                        f"Predicted sentiment: {prediction} 🙁"
                    )

                else:
                    st.info(
                        f"Predicted sentiment: {prediction} 😐"
                    )

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(features)[0]

                    probability_df = pd.DataFrame({
                        "Sentiment": model.classes_,
                        "Model Score": probabilities
                    }).sort_values(
                        "Model Score",
                        ascending=False
                    )

                    st.markdown("### Model Probability Estimates")

                    st.bar_chart(
                        probability_df.set_index("Sentiment")
                    )

                    st.caption(
                        "These are model probability estimates, not "
                        "guaranteed measures of real-world confidence."
                    )

                with st.expander("View cleaned text"):
                    st.write(cleaned)


# ==================================================
# 10. DATASET EXPLORER PAGE
# ==================================================
elif page == "Dataset Explorer":

    st.markdown("## Explore the Review Dataset")

    available_columns = list(df.columns)

    default_columns = [
        col for col in
        ["App", "Review_Date", "Star_Rating", "Review_Text", "Sentiment"]
        if col in available_columns
    ]

    selected_columns = st.multiselect(
        "Select columns to display",
        options=available_columns,
        default=default_columns
    )

    sentiment_filter = st.multiselect(
        "Filter by sentiment",
        options=sentiment_order,
        default=sentiment_order
    )

    filtered_df = df[
        df["Sentiment"].isin(sentiment_filter)
    ]

    search_text = st.text_input("Search review text")

    if search_text.strip():

        filtered_df = filtered_df[
            filtered_df["Review_Text"]
            .fillna("")
            .str.contains(
                search_text,
                case=False,
                regex=False
            )
        ]

    st.write(f"Matching reviews: {len(filtered_df):,}")

    if selected_columns:

        st.dataframe(
            filtered_df[selected_columns],
            use_container_width=True
        )

    else:
        st.info("Select at least one column to display.")

    st.download_button(
        label="Download Filtered Reviews",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="filtered_chatgpt_reviews.csv",
        mime="text/csv"
    )


# ==================================================
# 11. FOOTER
# ==================================================
st.divider()

st.caption(
    "Opinion Mining Case Study | Text Preprocessing, TF-IDF, "
    "Sentiment Classification, Dimensionality Reduction, "
    "and Visualization"
)
