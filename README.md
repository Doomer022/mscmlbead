# Fruit Classification using a Convolutional Neural Network (CNN)
This project implements a deep learning solution using PyTorch to classify images of various fruits. It utilizes a custom-built Convolutional Neural Network (CNN) architecture, incorporates advanced techniques like Data Augmentation, Hyperparameter Optimization (HPO) with Optuna, and custom training logic with Early Stopping to achieve high accuracy.

# 🌟 Project Features and Implementation
This project was developed following several advanced machine learning criteria, including:

Framework: Implemented entirely using PyTorch and torchvision.

Architecture: Custom Convolutional Neural Network (CNN) architecture.

Deep Model: The network design includes more than 7 hidden layers (Convolutional blocks, ReLU activations, Pooling layers, and Dense layers).

Special Layers: Includes multiple Dropout layers for regularization, which are essential to prevent overfitting during training on a large dataset.

Data Augmentation: Implements advanced augmentation techniques such as RandomRotation, RandomHorizontalFlip, ColorJitter, RandomPerspective, and the tensor-level augmentation transforms.RandomErasing.

Normalization: Input data is normalized using standard ImageNet means and standard deviations.

Hyperparameter Optimization (HPO): Uses the Optuna library to tune critical hyperparameters, including the number of convolutional layers, filter counts, dense unit counts, dropout rates, and the learning rate, maximizing validation accuracy.

Training Logic: Features custom-made, low-level training logic incorporating Early Stopping based on validation loss, ensuring the model converges efficiently without overfitting.

Evaluation: The performance is assessed using multiple metrics, including overall accuracy, Classification Report (Precision, Recall, F1-Score), and a visual Confusion Matrix.

# 🧠 Model Architecture (FruitCNN)
The custom CNN architecture is dynamically built and includes three main components:

Feature Extractor (self.features): A configurable number of convolutional blocks (Conv2D -> ReLU -> MaxPool2D). The HPO determines the optimal number of blocks (3-5) and the starting filter size (64, 128, or 256).

Flattening: Converts the 3D feature maps into a 1D vector.

Classifier (self.classifier): A sequence of fully connected (Dense) layers responsible for the final classification. It includes multiple Dropout layers to regularize the large number of parameters in the dense part of the network.

# 🚀 Setup and Installation
Prerequisites: Python 3.8+

A local virtual environment (venv) is recommended.

Installation:
The project uses a local virtual environment for dependency isolation.

Create the environment (Linux/macOS or Windows):
```python -m venv fruit_venv```

Activate (Linux/macOS):
```source fruit_venv/bin/activate```

Activate (Windows PowerShell):
```.\fruit_venv\Scripts\Activate.ps1```

Install the required libraries:
```pip install torch torchvision torchaudio optuna scikit-learn matplotlib seaborn kaggle```

Switch the Jupyter Kernel: In Visual Studio Code, ensure the active kernel for the notebook is set to the newly created fruit_venv.

# 💻 How to Run the Project
The entire workflow, including data download, HPO, training, and evaluation, is contained within the Jupyter Notebook.

Run the cells one after another, or run the whole thing at once.
