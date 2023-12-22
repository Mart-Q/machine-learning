# Machine Learning 

This endpoint provides a function for recipe recommendation according to user input. The user can send a food ingredient name to the API, 
and the API will return a recipe recommendation The API uses a pre-trained object detection model to recommend a suitable recipe.

The ML team consist of:
1. Rendra Baskoro Tuharea (M183BSY0890)
2. Mufidatul Ngazizah (M183BSX1681)
3. Muhammad Khatama Insani (M316BSY1231)
   
### API
We using Python Flask to build the API's that'll consume to MD and ML
[![N|Solid](https://vercel.com/_next/image?url=https%3A%2F%2Fimages.ctfassets.net%2Fe5382hct74si%2F6Dqa9T8XOOC95yJb0z9jew%2Fce4932b8d23046f260510e24c1ec39e1%2Fthumbnail.png&w=1920&q=75&dpl=dpl_8whFbfnjCmzPv538NhNbpsGCuH7g)](https://flask.palletsprojects.com/en/3.0.x/)

### How to Use
##### 1. Clone this repository
#
```
git clone https://github.com/Mart-Q/Backend-Python.git
```

##### 2. Install requirements
#
```
pip install -r requirements.txt
```

### MODEL Machine Learning
From Machine Learning using library Tensorflow, pickle, and joblib

#### Base URL:
|  https://us-central1-capstone-martq.cloudfunctions.net/dbmartq/

#### MODEL API:
```
POST {{host}}/dbmartq/recommender
```

Response:

```
   {
    "input_ingredients": [
        "gurame"
    ],
    "recommendations": [
        "gurame kecap",
        "gurame saus padang",
        "sup ikan gurame",
        "gurame crispy asam manis ala amih fatih",
        "sop ikan gurame"
    ]
}
```
And the rest of model for develop is:
- Daily Recommended recipe
- Recipe Generator



