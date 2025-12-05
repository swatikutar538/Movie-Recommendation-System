import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(movie_title):
    if movie_title not in movies["title"].values:
        return ["Movie not found in database"]

    # Get index of the movie
    idx = movies.index[movies["title"] == movie_title][0]
    
    # Get similarity scores
    scores = list(enumerate(similarity[idx]))
    
    # Sort movies by similarity (highest first)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Recommend top 5 (skip the movie itself)
    recommended = []
    for i in scores[1:6]:
        recommended.append(movies.iloc[i[0]]["title"])
    
    return recommended

# Program start
if __name__ == "__main__":
    user_movie = input("Enter a movie title: ")
    print("\nRecommended Movies:")
    for m in recommend(user_movie):
        print("- " + m)
