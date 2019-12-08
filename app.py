from io import BytesIO
import json
import torch.nn as nn
import pickle
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from CyrillicDataset import ImagePredictor
from torch.utils.data import Dataset, DataLoader
import pretrainedmodels
app = Flask(__name__)

RESCALE_SIZE = 278
MODEL_NAME="ALEX"

if MODEL_NAME=="VGG":
  model_mixed=models.vgg19()
elif MODEL_NAME=="XCEPTION":
  model_mixed = pretrainedmodels.__dict__['xception'](num_classes=1000)
else:
  model_mixed = models.alexnet(pretrained=False)

n_classes = 33
# Заменяем Fully-Connected слой на наш линейный классификатор
if MODEL_NAME!="XCEPTION":
  num_features = model_mixed.classifier[-1].in_features
  features = list(model_mixed.classifier.children())[:-1] # Remove last layer
  features.extend([nn.Linear(num_features, 33)]) # Add our layer with n outputs
  model_mixed.classifier = nn.Sequential(*features) # Replace the model classifier
else:
  model_mixed.last_linear=nn.Linear(in_features=2048, out_features=33, bias=True)
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
if MODEL_NAME=="VGG":
  model_mixed.load_state_dict(torch.load('vgg19.path'))
elif MODEL_NAME=="XCEPTION":
  model_mixed.load_state_dict(torch.load('xception.path'))
else:
  model_mixed.load_state_dict(torch.load('alex_mixed.path'))
model_mixed=model_mixed.cuda()
model_mixed.eval()
DEVICE = torch.device("cuda")
def get_prediction_from_img(img):
    test_dataset = ImagePredictor([img], mode="test")
    test_loader = DataLoader(test_dataset)
    probs = predict(model_mixed, test_loader)
    return [{'letter':label_encoder.inverse_transform([i])[0], 'prob':str(x)} for i,x in enumerate(probs[-1])]

def predict(model, test_loader):
    with torch.no_grad():
        logits = []
    
        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
 
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
            
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs

@app.route('/', methods=['GET'])
def test():
    return 'Everything works fine'

@app.route('/predict', methods=['POST'])
def get_prediction():
    img=BytesIO(request.files['photo'].read())
    print(get_prediction_from_img(img))
    return jsonify(get_prediction_from_img(img))

if __name__ == '__main__':
    app.run()
