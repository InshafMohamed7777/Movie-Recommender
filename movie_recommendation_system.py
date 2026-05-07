# Import necessary libraries for data manipulation, cosine similarity computation, and collaborative filtering
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Load the dataset (use the MovieLens 100k dataset for simplicity)
# Assuming you have 'movies.csv' and 'ratings.csv' files extracted from the MovieLens dataset

# Step 1: Content-Based Filtering

# Load movies dataset, containing movie IDs, titles, and genres
movies = pd.read_csv('movies.csv')

# Create a new column to combine different "content" features (for simplicity, we use genres)
movies['content'] = movies['genres']

# Use TF-IDF Vectorizer to convert genres/content into feature vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Calculate cosine similarity matrix between all movies based on their content features
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define a function to get movie recommendations using content-based filtering
def content_based_recommendations(title, cosine_sim, movies, top_n=10):
    # Get the index of the given movie title
    idx = movies[movies['title'] == title].index[0]
    
    # Get the pairwise similarity scores for this movie with all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top N similar movies
    movie_indices = [i[0] for i in sim_scores[1:top_n + 1]]
    
    # Return the titles of the top N recommended movies
    return movies['title'].iloc[movie_indices]

# Test content-based recommendation function
print("Content-based recommendations for 'Toy Story (1995)':")
print(content_based_recommendations('Toy Story (1995)', cosine_sim, movies))

# Step 2: Collaborative Filtering using Surprise Library (SVD)

# Load ratings dataset, containing user IDs, movie IDs, and ratings
ratings = pd.read_csv('ratings.csv')

# Load the dataset into Surprise using the Reader class
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train the SVD model and perform cross-validation
svd = SVD()  # Singular Value Decomposition for Collaborative Filtering
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the SVD model on the full dataset
trainset = data.build_full_trainset()
svd.fit(trainset)

# Define a function to get movie recommendations for a given user
def collaborative_filtering_recommendations(user_id, ratings, movies, svd, top_n=10):
    # Get all movie IDs
    all_movie_ids = movies['movieId'].unique()
    
    # Get the movies already rated by the user
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    
    # Predict ratings for all unrated movies
    recommendations = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movies:
            pred = svd.predict(user_id, movie_id)
            recommendations.append((movie_id, pred.est))
    
    # Sort recommendations by predicted rating in descending order
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    # Get the top N recommended movie IDs
    top_movie_ids = [rec[0] for rec in recommendations[:top_n]]
    
    # Return the titles of the top N recommended movies
    return movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()

# Test collaborative filtering recommendation function
print("Collaborative filtering recommendations for user 1:")
print(collaborative_filtering_recommendations(1, ratings, movies, svd))

# Step 3: Hybrid Recommendation System

# Define a simple hybrid function that combines both approaches
def hybrid_recommendations(user_id, movie_title, ratings, movies, cosine_sim, svd, top_n=10):
    # Get content-based recommendations
    content_recs = content_based_recommendations(movie_title, cosine_sim, movies, top_n=top_n * 2)
    
    # Get collaborative filtering recommendations
    collaborative_recs = collaborative_filtering_recommendations(user_id, ratings, movies, svd, top_n=top_n * 2)
    
    # Combine both lists and maintain unique movie titles
    combined_recs = list(set(content_recs + collaborative_recs))
    
    # Limit the combined recommendations to the top N unique movies
    return combined_recs[:top_n]

# Test the hybrid recommendation function
print("Hybrid recommendations for user 1 and 'Toy Story (1995)':")
print(hybrid_recommendations(1, 'Toy Story (1995)', ratings, movies, cosine_sim, svd))