import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import tkinter as tk
from tkinter import messagebox

# Load the movies dataset
movies = pd.read_csv('movies.csv')  # Replace with the correct path to your file

# Clean up the titles by removing special characters
def clean_title(title):
    return re.sub(r"[^a-zA-Z0-9 ]", "", title)

# Apply cleaning to the movie titles
movies['clean_title'] = movies['title'].apply(clean_title)

# Set up the TF-IDF Vectorizer and fit it to the cleaned titles
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies['clean_title'])

# Function to get recommendations based on cosine similarity
def get_recommendations(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    print("Similarity Scores:", similarity)  # Debugging output
    indices = np.argsort(similarity, axis=0)[-5:][::-1]
    print("Selected Indices:", indices)      # Debugging output

    results = movies.iloc[indices][['title', 'genres']]
    print("Results:", results)               # Debugging output

    return results

# Function to display recommendations in a popup
def show_recommendations():
    title = entry.get()
    recommendations = get_recommendations(title)
    if recommendations.empty:
        messagebox.showinfo("Recommendations", "No similar movies found.")
    else:
        result_text = "\n".join([f"{row['title']} - {row['genres']}" for _, row in recommendations.iterrows()])
        messagebox.showinfo("Recommendations", result_text)

# Setting up the Tkinter GUI
root = tk.Tk()
root.title("Movie Recommendation System")

label = tk.Label(root, text="Enter Movie Title:")
label.pack(padx=10, pady=5)

entry = tk.Entry(root, width=40)
entry.pack(padx=10, pady=5)

search_button = tk.Button(root, text="Get Recommendations", command=show_recommendations)
search_button.pack(pady=10)

root.mainloop()
