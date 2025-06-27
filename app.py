from flask import Flask, request, jsonify
import torch
from model import CNN36  # your model class
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)
model = CNN36()
model.load_state_dict(torch.load("cnn36_model.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('L')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, 1).item()

    if pred <= 9:
        result = str(pred)
    else:
        result = chr(pred - 10 + ord('A'))

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run()
