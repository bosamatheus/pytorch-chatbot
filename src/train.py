import os
import json

import numpy as np

from torch import (
    nn,
    utils.data.DataLoader as DataLoader,
    device as torch_device,
    cuda as torch_cuda,
    optim.Adam as Adam,
    save as torch_save
)


from nltk_utils import tokenize, stem, bag_of_words
from chat_dataset import ChatDataset
from model import NeuralNet


# Loading our JSON file with intents
with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json'), 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Looping through our data
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", "!", ".", ",", "'"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) # get unique words
tags = sorted(set(tags)) # get unique tags

X_train = []
y_train = []
# Mounting our train data
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss (PyTorch) or One-Hot encode

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0]) #len(all_words)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch_device('cuda' if torch_cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Our training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels.long())

        # Backward and optimizer steps
        optimizer.zero_grad()
        loss.backward() # calculate backpropagation
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}')

print(f'Final, Loss={loss.item():.4f}')

# Save our model
data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags
}

FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.pth')
torch_save(data, FILE)

print(f'Training completed. File saved to {FILE}')
