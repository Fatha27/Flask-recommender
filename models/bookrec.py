# book_recommender.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
class BookRecommender:
    def __init__(self):
        # Load and preprocess the dataset
        self.data = self.load_and_preprocess_data()

        # Calculate the similarity matrix
        self.similarity_matrix = self.calculate_similarity_matrix()

    def load_and_preprocess_data(self):

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Combine the script directory with the filename to get the full path
        file_path = os.path.join(script_dir, 'Goodreads_BestBooksEver_1-10000.csv')
        bdf = pd.read_csv(file_path, engine='python')

        # Load the dataset
        #bdf = pd.read_csv('Goodreads_BestBooksEver_1-10000.csv')

        # Drop unnecessary columns
        bdf.drop('url', axis=1, inplace=True)

        # Drop rows with missing values
        bdf = bdf.dropna()

        # Reset the index
        bdf = bdf.reset_index(drop=True)

        # Select relevant columns
        bdfc = bdf[['bookISBN', 'bookTitle', 'bookAuthors', 'bookDesc', 'bookGenres']]

        # Split bookGenres and keep only the first three genres
        bdfc['bookGenres'] = bdfc['bookGenres'].apply(lambda x: x.split('|')[:3])

        # Function to extract only alphabets from a string
        def extract_alphabets(genres_list):
            import re
            return [re.sub(r'[^a-zA-Z\s]', '', genre.split('/')[0]) for genre in genres_list]

        # Apply the function to the 'bookGenres' column
        bdfc['CleanedGenres'] = bdfc['bookGenres'].apply(extract_alphabets)

        # Remove spaces from bookAuthors and bookGenres columns
        bdfc['bookAuthors'] = bdfc['bookAuthors'].str.replace(' ', '')
        bdfc['ClGenres'] = bdfc['CleanedGenres'].apply(lambda genres: [genre.replace(' ', '') for genre in genres])

        # Drop unnecessary columns
        bdfc.drop('bookGenres', axis=1, inplace=True)

# Extract the part before the first comma in 'bookAuthors'
        bdfc['CleanedAuthors'] = bdfc['bookAuthors'].str.split(',').str[0]
        # Drop unnecessary columns
        bdfc.drop(['bookAuthors', 'CleanedGenres'], axis=1, inplace=True)

        # Split bookDesc into words
        bdfc['Desc'] = bdfc['bookDesc'].apply(lambda x: x.split())

        # Create a list with a single-element list for each author
        bdfc['Author'] = bdfc['CleanedAuthors'].apply(lambda x: [x])

        # Drop unnecessary columns
        bdfc.drop(['CleanedAuthors', 'bookDesc'], axis=1, inplace=True)

        # Combine ClGenres, Desc, and Author into a single column
        bdfc['fulltag'] = bdfc['ClGenres'] + bdfc['Desc'] + bdfc['Author']

        # Convert the lists to strings
        bdfc['fulltag'] = bdfc['fulltag'].apply(lambda x: " ".join(x))

        # Drop unnecessary columns
        finbd = bdfc.drop(columns=['ClGenres', 'Desc', 'Author'])

        return finbd

    def calculate_similarity_matrix(self):
        from sklearn.feature_extraction.text import CountVectorizer

        # Create a CountVectorizer
        cv = CountVectorizer(max_features=5000, stop_words='english')

        # Transform the 'fulltag' column into a bag-of-words
        vector = cv.fit_transform(self.data['fulltag']).toarray()

        # Calculate cosine similarity matrix
        similarity = cosine_similarity(vector)

        return similarity

    def recommend_books(self, book):
        # Find the index of the book
        index = self.data[self.data['bookTitle'] == book].index[0]

        # Calculate cosine similarity distances
        distances = sorted(list(enumerate(self.similarity_matrix[index])), reverse=True, key=lambda x: x[1])

        # Display the top 5 recommended books
        recommended_books = [self.data.iloc[i[0]].bookTitle for i in distances[1:6]]
        return recommended_books

#if __name__ == '__main__':
    # Create an instance of BookRecommender
 #   book_recommender = BookRecommender()

    # Example recommendation
  #  book_recommendations = book_recommender.recommend_books('Murder on the Orient Express')
   # print(f"Top 5 books similar to 'Murder on the Orient Express': {book_recommendations}")
