from flask import Flask, render_template, request, jsonify
import json
import joblib
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader

app = Flask(__name__)

with open('svd_model.pkl', 'rb') as pickle_file:
    svd_model = joblib.load(pickle_file)

file_path = r'D:\myproject\dataset_daily.csv'
resep_df = pd.read_csv(file_path)

def recommend_recipe(user_id, svd_model, resep_df, trainset):
    recipes_to_recommend = []

    for item_id in resep_df['Title'].unique():
        if not trainset.knows_user(user_id) or not trainset.knows_item(item_id):
            recipes_to_recommend.append(item_id)

    return recipes_to_recommend[:5]

reader = Reader(rating_scale=(1, 500))
dataset = Dataset.load_from_df(resep_df[['User', 'Title', 'Loves']], reader)
trainset = dataset.build_full_trainset()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['user_id']
        recommendations = recommend_recipe(user_id, svd_model, resep_df, trainset)
        return jsonify({'user_id': user_id, 'recommendations': recommendations})
    return render_template('daily.html')

if __name__ == '__main__':
    app.run(debug=True)
