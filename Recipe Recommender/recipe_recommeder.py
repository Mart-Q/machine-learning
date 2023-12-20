from flask import Flask, render_template, request, jsonify, Response
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import json

app = Flask(__name__)

with open('recipe_recommendation.pkl', 'rb') as model_file:
    cosine_sim = pickle.load(model_file)

df = pd.read_csv("dataset_recipe.csv")

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined'])

def recommend_recipe(selected_ingredients, df, cosine_sim=cosine_sim):
    selected_text = ' '.join(selected_ingredients)
    df_temp = df.copy()
    df_temp['selected'] = selected_text
    df_temp['combined_selected'] = df_temp['Title'] + ' ' + df_temp['selected']

    tfidf_matrix_selected = tfidf_vectorizer.transform(df_temp['combined_selected'])
    cosine_sim_selected = linear_kernel(tfidf_matrix_selected, tfidf_matrix)
    recommended_indices = cosine_sim_selected[0].argsort()[:-6:-1]

    filtered_indices = [idx for idx in recommended_indices if selected_text.lower() in df['Title'].iloc[idx].lower()]
    remaining_recommendations = 5 - len(filtered_indices)

    additional_indices = [idx for idx in recommended_indices if idx not in filtered_indices][:remaining_recommendations]
    filtered_indices += additional_indices

    recommended_recipes = df['Title'].iloc[filtered_indices]

    return recommended_recipes.tolist()

@app.route('/')
def index():
    return render_template('mvp.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        selected_ingredients = request.form.get('ingredients', '').split(',')
        recommendations = recommend_recipe(selected_ingredients, df)

        return render_template('recipe.html', input_ingredients=selected_ingredients, recommendations=recommendations)

    except Exception as e:
        return render_template('recipe.html', error_message=str(e))

@app.route('/download_json', methods=['GET'])
def download_json():
    try:
        selected_ingredients = request.args.get('ingredients', '').split(',')
        recommendations = recommend_recipe(selected_ingredients, df)

        json_data = {
            'input_ingredients': selected_ingredients,
            'recommendations': recommendations
        }

        response = Response(
            json.dumps(json_data, indent=2),
            content_type='application/json'
        )

        response.headers["Content-Disposition"] = "attachment; filename=recommendations.json"

        return response

    except Exception as e:
        return render_template('recipe.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
