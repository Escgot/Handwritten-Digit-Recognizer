# ✏️ Handwritten Digit Recognizer

A Convolutional Neural Network (CNN) that recognizes handwritten digits (0–9) trained on the MNIST dataset.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~99% |
| Model | CNN |
| Dataset | MNIST (60k train / 10k test) |

---

## 🧠 Model Architecture

```
Input (28×28×1)
    ↓
Conv2D(32) + BatchNorm + MaxPooling
    ↓
Conv2D(64) + BatchNorm + MaxPooling
    ↓
Flatten
    ↓
Dense(128) + Dropout(0.4)
    ↓
Dense(10) → Softmax
```

---

## ✨ Features

- CNN with 2 convolutional blocks
- BatchNormalization for faster, stable training
- Early stopping to prevent overfitting
- Confusion matrix + classification report
- Visualizes images the model got wrong
- Interactive Gradio sketchpad — draw a digit and get a live prediction

---

## 🚀 How to Run

1. Open `mnist_digit_recognizer.ipynb` in [Google Colab](https://colab.research.google.com)
2. Runtime → Run all (`Ctrl+F9`)
3. Training takes ~3 minutes on the free GPU
4. Draw digits in the Gradio sketchpad in the last cell

---

## 🛠️ Tech Stack

Python · TensorFlow · Keras · Gradio · NumPy · Matplotlib · scikit-learn

---

## 👤 Author

**[Mohamed Ouledali]** — First-year Engineering Student
