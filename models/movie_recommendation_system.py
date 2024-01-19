# movie_recommender.py
import os
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Combine the script directory with the filename to get the full path
        file_path = os.path.join(script_dir, 'moviesd.csv')

        # loading the data from the csv file to a pandas dataframe
        self.movies_data = pd.read_csv(file_path, engine='python')

        # selecting the relevant features for recommendation
        selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

        # replacing the null values with null string
        for feature in selected_features:
            self.movies_data[feature] = self.movies_data[feature].fillna('')

        # combining all the 5 selected features
        combined_features = self.movies_data['genres'] + ' ' + self.movies_data['keywords'] + ' ' + self.movies_data[
            'tagline'] + ' ' + self.movies_data['cast'] + ' ' + self.movies_data['director']

        # converting the text data to feature vectors
        self.vectorizer = TfidfVectorizer()
        self.feature_vectors = self.vectorizer.fit_transform(combined_features)

        # getting the similarity scores using cosine similarity
        self.similarity = cosine_similarity(self.feature_vectors)

    def recommend_movies(self, movie_name):
        list_of_all_titles = self.movies_data['title'].tolist()

        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if not find_close_match:
            return []  # No close match found

        close_match = find_close_match[0]

        index_of_the_movie = self.movies_data[self.movies_data.title == close_match]['index'].values[0]

        similarity_score = list(enumerate(self.similarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        recommended_movies = []
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = self.movies_data[self.movies_data.index == index]['title'].values[0]
            recommended_movies.append(title_from_index)

        return recommended_movies[1:7]  # Return top 10 recommended movies
