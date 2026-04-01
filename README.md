# 🧠 Handwritten Digit Recognition using Neural Networks

## 📌 Project Overview

This project focuses on building a **machine learning + deep learning system** to recognize handwritten digits (0–9).

It uses the famous **MNIST dataset**, where each digit is represented as a **28×28 grayscale image**.

The project demonstrates:

* Traditional Machine Learning models
* Neural Network (Deep Learning)
* Model evaluation using multiple metrics
* Real-time digit recognition using drawing interface

---

## 🚀 Features

✔ Multiple ML models (Logistic Regression, Random Forest, KNN)
✔ Deep Learning model using Neural Networks
✔ Accuracy, Precision, Recall, F1-score
✔ Confusion Matrix visualization
✔ Training graphs (Accuracy & Loss)
✔ Model comparison graph
✔ Real-time digit prediction (draw using mouse)

---

## 📂 Dataset

* Dataset used: **MNIST (built-in from TensorFlow)**
* Contains:

  * 60,000 training images
  * 10,000 testing images
* Each image:

  * Size: **28 × 28 pixels**
  * Grayscale (0–255)

---

## ⚙️ Technologies Used

* Python
* NumPy
* Matplotlib
* OpenCV
* Scikit-learn
* TensorFlow / Keras

---

## 🧪 Models Used

### 🔹 Machine Learning Models

* Logistic Regression
* Random Forest
* K-Nearest Neighbors (KNN)

### 🔹 Deep Learning Model

* Neural Network with:

  * Input layer (784 neurons)
  * Hidden layers (128, 64 neurons)
  * Output layer (10 classes)

---

## 📊 Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📈 Output Visualizations

The project generates:

* Confusion Matrix for each model
* Accuracy & Loss graphs for Neural Network
* Model comparison bar graph
* Sample predictions

---

## ✏️ Live Digit Drawing Feature

You can draw a digit using your mouse and the model will predict it.

### Controls:

* `p` → Predict digit
* `c` → Clear screen
* `q` → Quit

---

## 🛠️ Installation

Run this command in terminal:

```bash
pip install numpy matplotlib opencv-python scikit-learn tensorflow
```

---

## ▶️ How to Run

```bash
python ml.py
```

---

## ⚡ Project Workflow

1. Load dataset (MNIST)
2. Preprocess data (reshape + normalize)
3. Train ML models
4. Evaluate models
5. Train Neural Network
6. Generate graphs
7. Compare models
8. Draw and predict digit in real-time

---

## 📌 Results

* Machine Learning models give decent accuracy (~90–95%)
* Neural Network gives higher accuracy (~97–98%)
* Real-time prediction works based on drawing quality

---

## ⚠️ Limitations

* Accuracy depends on how clearly you draw digits
* Model trained on MNIST → may not generalize perfectly to all handwriting styles

---

## 💡 Future Improvements

* Use CNN (Convolutional Neural Network) for better accuracy
* Add GUI using Tkinter or Web App
* Use custom dataset for better real-world performance

---

## 👨‍💻 Author

Developed as part of internship project on:
**Handwritten Digit Recognition using Neural Networks**

---

## ⭐ Conclusion

This project helps understand:

* Basics of Machine Learning
* Deep Learning concepts
* Image classification
* Real-world application using computer vision

---
