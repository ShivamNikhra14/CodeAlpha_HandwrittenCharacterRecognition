# CodeAlpha_HandwrittenCharacterRecognition

# 🧠 Handwritten Character Recognition (Digits + Letters)

This project implements a deep learning model using **PyTorch** to recognize **handwritten digits (0–9)** from the **MNIST** dataset and **uppercase letters (A–Z)** from the **EMNIST Letters** dataset.

## 📌 Objective

To build a Convolutional Neural Network (CNN) that accurately classifies grayscale handwritten characters, combining digits and letters into a unified model.

## 🛠️ Features

- 📦 Supports both **MNIST** and **EMNIST (Letters)** datasets.
- 🧠 Uses **Convolutional Neural Networks (CNN)** for feature extraction.
- 🔁 Combines the two datasets and trains a single model to recognize **36 classes** (0–9, A–Z).
- 📊 Includes training and test accuracy, and a confusion matrix for evaluation.
- 📈 Visualizes predictions and model performance.

## 📚 Datasets

| Dataset | Classes     | Size                        |
|---------|-------------|-----------------------------|
| MNIST   | 0–9         | 60,000 train / 10,000 test  |
| EMNIST  | A–Z (uppercase) | 145,600 train / 14,400 test |

## 📦 Requirements

- Python 3.10+
- PyTorch
- torchvision
- matplotlib

Install dependencies with:
pip install torch torchvision matplotlib

## 🚀 How to Run

Run the full training script:
python main.py

Or open the notebook version in Jupyter:
jupyter notebook Handwritten_Character_Recognition.ipynb

## 🧠 Model Architecture

The CNN model consists of:

- 2 × Convolutional layers + ReLU + MaxPooling
- Flatten layer
- 2 × Fully connected (Linear) layers
- Softmax output for 36 classes

## 📈 Sample Output

- Test Accuracy: ~95–97% on MNIST, ~85–90% on EMNIST letters
- Confusion Matrix visualization
- Random predictions visualization with labels

## 📦 Folder Structure
.
├── main.py # All-in-one training + evaluation script
├── README.md # Project documentation
├── assets/ # (Optional) Output visualizations

## ✍️ Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- PyTorch community

## 🧩 Future Improvements

- ✅ Add support for lowercase letters
- 📝 Extend to word or sentence-level recognition using CRNN
- 🖼️ Build a web interface for real-time predictions