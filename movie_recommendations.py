# Import required libraries for data handling, numerical operations, machine learning, text processing, and GUI development
import pandas as pd  # For handling data in DataFrames
import numpy as np  # For numerical operations, including sorting and indexing
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical form (TF-IDF)
from sklearn.metrics.pairwise import cosine_similarity  # For measuring similarity between vectors
import re  # For regular expression operations, used to clean text
import tkinter as tk  # For building the GUI application
from tkinter import messagebox  # For showing popup messages in Tkinter

# Load the movies dataset with columns like movieId, title, and genres
movies = pd.read_csv('movies.csv')  # Ensure 'movies.csv' file is in the same directory or provide the correct path

# Load the ratings dataset with columns like userId, movieId, and rating
ratings = pd.read_csv('ratings.csv')  # Ensure 'ratings.csv' file is in the same directory or provide the correct path

# Function to clean movie titles by removing special characters, leaving only letters, numbers, and spaces
def clean_title(title):
    return re.sub(r"[^a-zA-Z0-9 ]", "", title)  # Replace any non-alphanumeric characters with an empty string

# Apply the clean_title function to each title in the movies dataset, creating a new column 'clean_title'
movies['clean_title'] = movies['title'].apply(clean_title)

# Initialize a TF-IDF Vectorizer, which converts movie titles into a numerical matrix representation
# ngram_range=(1, 2) means it considers single words and pairs of consecutive words (bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Fit the vectorizer to the cleaned titles and transform them into a TF-IDF matrix
# This matrix represents the importance of words in each title
tfidf = vectorizer.fit_transform(movies['clean_title'])

# Function to find similar movies based on TF-IDF cosine similarity, given a search title
def search(title):
    title = clean_title(title)  # Clean the input title using the clean_title function
    query_vec = vectorizer.transform([title])  # Transform the cleaned title to a TF-IDF vector
    similarity = cosine_similarity(query_vec, tfidf).flatten()  # Compute cosine similarity between the query and all titles
    indices = np.argsort(similarity, axis=0)[-5:][::-1]  # Get indices of the top 5 most similar movies
    results = movies.iloc[indices][['movieId', 'title', 'genres']]  # Retrieve titles and genres of similar movies
    return results  # Return the top 5 similar movies as a DataFrame

# Collaborative filtering function to find recommended movies based on user ratings for a given movie ID
def find_similar_movies(movie_id):
    # Find users who rated the specified movie highly (rating > 4) and get their unique user IDs
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()

    # Find all movies rated highly by these similar users
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]

    # Calculate the percentage of similar users who rated each recommended movie
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    # Filter to retain only movies rated highly by more than 10% of similar users
    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    # Find all users who rated these recommended movies highly
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]

    # Calculate the percentage of all users who rated each recommended movie
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    # Combine similar_user_recs and all_user_recs to calculate the recommendation score
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]  # Rename columns for clarity

    # Calculate a score by dividing the similar percentage by the all percentage for each movie
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    # Sort recommendations by score in descending order
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    # Merge with movies dataset to get the title and genre information for the top 10 recommendations
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["title", "genres", "score"]]

# Function to handle user input, retrieve recommendations, and display them in the GUI
def show_recommendations():
    title = entry.get()  # Get the text input from the entry widget
    search_results = search(title)  # Use the search function to find movies similar to the input title
    
    # If no similar movies are found in the search results, show a popup message
    if search_results.empty:
        messagebox.showinfo("Recommendations", "No similar movies found.")
    else:
        # Take the movieId of the first search result to use for collaborative filtering recommendations
        movie_id = search_results.iloc[0]["movieId"]
        recommendations = find_similar_movies(movie_id)  # Get recommendations based on collaborative filtering
        
        # Clear any previous recommendations in the output text widget
        output_text.delete("1.0", tk.END)

        # If no collaborative recommendations are found, show a message in the output widget
        if recommendations.empty:
            output_text.insert(tk.END, "No collaborative recommendations found.\n")
        else:
            # Display each recommendation with title, genre, and score in the output widget
            for _, row in recommendations.iterrows():
                output_text.insert(tk.END, f"{row['title']} - {row['genres']} (Score: {row['score']:.2f})\n")

# Setting up the Tkinter GUI
root = tk.Tk()  # Create the main window
root.title("Movie Recommendation System")  # Set the title of the window

# Label for the input field
label = tk.Label(root, text="Enter Movie Title:")  # Create a label for the input box
label.pack(padx=10, pady=5)  # Add padding around the label

# Text entry box for movie title input
entry = tk.Entry(root, width=40)  # Create an entry widget for user to type the movie title
entry.pack(padx=10, pady=5)  # Add padding around the entry box

# Search button that triggers the show_recommendations function
search_button = tk.Button(root, text="Get Recommendations", command=show_recommendations)  # Create a button to get recommendations
search_button.pack(pady=10)  # Add vertical padding around the button

# Text widget to display the recommendations
output_text = tk.Text(root, height=15, width=50, wrap="word")  # Create a text widget to show recommendations
output_text.pack(padx=10, pady=10)  # Add padding around the output text widget

# Run the Tkinter main loop to display the GUI and wait for user interaction
root.mainloop()
