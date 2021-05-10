import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_features=4, layer1=10, layer2=20, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, out_features)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
    def save_model(self, model, PATH:str):
        try:
            torch.save(model.state_dict(), "../model/" + PATH)
        except FileNotFoundError:
            print ("Invalid file! Make sure model directory exists and is given the right write access.")
        finally:
            print("Model saved!")