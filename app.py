from flask import (
    Flask, 
    render_template, 
    request, 
    jsonify, 
    make_response
)
from src.chat import get_answer


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    answer, prob = get_answer(sentence=req.get('message'))
    print(f'Probability={prob:.4f}')
    res = make_response(jsonify(answer), 200)
    return res
