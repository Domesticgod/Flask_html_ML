from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__, template_folder='templates')

# Load the model and data
with open('model_data.pkl', 'rb') as file:
    model_data = pickle.load(file)

user_item_matrix = model_data['user_item_matrix']
user_similarity_df = model_data['user_similarity_df']


def get_song_recommendations(input_songs, user_item_matrix, user_similarity_df, top_n=10):
    # Get the user preferences based on the input song
    user_preferences = user_item_matrix.iloc[:, input_songs]

    # Calculate weighted song recommendations using user similarity
    weighted_recommendations = user_similarity_df.dot(user_preferences)

    # Exclude the input song from the list of recommended songs
    weighted_recommendations = weighted_recommendations.drop(user_item_matrix.columns[input_songs], errors='ignore')

    # Get top N recommendations
    top_recommendations = weighted_recommendations.nlargest(top_n)

    return top_recommendations.index.tolist()


# Define routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    input_songs = int(request.form.get('input_songs'))
    # input_songs = input_songs.split(',')  # Assuming input songs are comma-separated

    recommendations = get_song_recommendations(input_songs, user_item_matrix, user_similarity_df, top_n=10)

    return render_template('recommendations.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
