import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie dataset
movies_df = pd.read_csv("movies.csv")

# Create a TfidfVectorizer object
tfidf_vectorizer = TfidfVectorizer()

# Transform the movie descriptions into vectors
movie_vectors = tfidf_vectorizer.fit_transform(movies_df["description"])

# Calculate the cosine similarity between all pairs of movies
movie_similarity = cosine_similarity(movie_vectors)

# Function to recommend movies
def recommend_movies(movie_name):
  """
  Recommends movies similar to the given movie name.

  Args:
    movie_name: The name of the movie to recommend similar movies to.

  Returns:
    A list of movie names that are similar to the given movie name.
  """

  # Find the index of the movie in the dataset
  movie_index = movies_df[movies_df["title"] == movie_name].index[0]

  # Get the similarity scores for all movies
  similarity_scores = movie_similarity[movie_index, :]

  # Sort the movies by their similarity scores
  sorted_movie_indices = similarity_scores.argsort()[::-1]

  # Get the top 10 most similar movies
  top_10_movie_indices = sorted_movie_indices[:10]

  # Get the names of the top 10 most similar movies
  recommended_movies = movies_df.iloc[top_10_movie_indices]["title"].tolist()

  return recommended_movies

# Get the user's input
movie_name = input("Enter a movie name: ")

# Recommend movies similar to the user's input
recommended_movies = recommend_movies(movie_name)

# Print the recommended movies
print("Here are some movies similar to {}:".format(movie_name))
print(recommended_movies)