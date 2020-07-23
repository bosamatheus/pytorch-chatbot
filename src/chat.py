import os
import random
import json

from torch import (
    load as torch_load,
    device as torch_device,
    cuda as torch_cuda,
    from_numpy as torch_from_numpy, 
    max as torch_max,
    softmax as torch_softmax,
) 

from src.nltk_utils import bag_of_words, tokenize
from src.model import NeuralNet


def get_answer(sentence):
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json'), 'r') as f:
        intents = json.load(f)

    FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.pth')
    data = torch_load(FILE)

    input_size = data['input_size']
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    all_words = data['all_words']
    tags = data['tags']
    model_state = data['model_state']

    device = torch_device('cuda' if torch_cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch_from_numpy(X)

    output = model(X)
    _, predicted = torch_max(output, dim=1)
    tag = tags[predicted.item()]

    answer = ''
    probs = torch_softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                answer = random.choice(intent['responses'])
    else:
        answer = "I don't understand..."
    
    return answer, prob
