import pandas as pd

# Load the downloaded dataset
df = pd.read_csv(
    "The_ Generative_AI_Ecosystem_50k_User_Reviews_2026.csv"
)

# Display basic information
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five reviews:")
print(df.head().to_string())

print("\nMissing values:")
print(df.isnull().sum())

print("\nApplications in the dataset:")
print(df["App"].value_counts())

print("\nStar rating distribution:")
print(df["Star_Rating"].value_counts().sort_index())

print("\nReview themes:")
print(df["Review_Theme"].value_counts())