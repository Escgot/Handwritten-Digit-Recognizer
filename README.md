# ✏️ Handwritten Digit Recognizer

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-FFD21E?logo=gradio&logoColor=black)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

An optimized deep learning pipeline implementing a robust Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset. By incorporating real-time data augmentation layers, batch normalization, and a dynamic learning rate schedule, this system bridges the gap between clean lab training environments and noisy, real-world sketching styles.

---

## 📊 Performance Benchmark

| Evaluation Metric | Score | Implementation Strategy |
| :--- | :--- | :--- |
| **Test Accuracy** | **~99.4% – 99.6%** | VGG-style Triple-Stacked Convolutions |
| **Final Test Loss** | **< 0.025** | Categorical Cross-Entropy |
| **Training Duration** | ~3–4 Minutes | GPU Accelerated (T4 Runtime) |
| **Invariance Scale** | High Resilience | On-The-Fly Rotation, Zoom & Shift Layers |

---

## 🧠 Model Architecture

```text
        [ Input: 28×28×1 Grayscale ]
                     │
   [ Training Only: Augmentation Layers ]
   ├── Random Rotation  (±8%)
   ├── Random Zoom      (±8%)
   └── Random Translation (±8%)
                     │
   ┌─────────────────────────────────────┐
   │ Conv Block 1                        │
   │ ├── Conv2D (32, 3×3, ReLU, Same)    │
   │ ├── BatchNormalization              │
   │ ├── Conv2D (32, 3×3, ReLU, Same)    │
   │ ├── BatchNormalization              │
   │ └── MaxPooling2D + Dropout (0.25)   │
   └──────────────────┬──────────────────┘
                      │
   ┌─────────────────────────────────────┐
   │ Conv Block 2                        │
   │ ├── Conv2D (64, 3×3, ReLU, Same)    │
   │ ├── BatchNormalization              │
   │ ├── Conv2D (64, 3×3, ReLU, Same)    │
   │ ├── BatchNormalization              │
   │ └── MaxPooling2D + Dropout (0.25)   │
   └──────────────────┬──────────────────┘
                      │
   ┌─────────────────────────────────────┐
   │ Conv Block 3                        │
   │ ├── Conv2D (128, 3×3, ReLU, Same)   │
   │ ├── BatchNormalization              │
   │ └── Dropout (0.25)                  │
   └──────────────────┬──────────────────┘
                      │
   ┌─────────────────────────────────────┐
   │ Classification Head                 │
   │ ├── Flatten                         │
   │ ├── Dense (256, ReLU)               │
   │ ├── BatchNormalization              │
   │ ├── Dropout (0.5)                   │
   │ └── Dense (10, Softmax)             │
   └─────────────────────────────────────┘
```

---

## ✨ Core Features

- **In-Graph Spatial Augmentation:** Rotation, zoom, and translation layers are built directly into the model graph. Active only during training, they force the network to learn invariant geometric structures — making it robust to off-center or tilted drawings.
- **Triple-Stacked Conv Topology:** Three convolutional blocks (32 → 64 → 128 filters) progressively extract edges, curves, and full digit structures before classification.
- **Batch Normalization:** Applied after every convolutional step to stabilize gradient distributions and accelerate convergence.
- **Image Inversion in Gradio:** MNIST trains on white digits on a black background. The Gradio sketchpad draws black strokes on white. The preprocessing pipeline inverts the canvas before inference — without this, predictions are garbage.
- **Dynamic LR Scheduling (`ReduceLROnPlateau`):** Halves the learning rate (factor = 0.5) after 2 epochs without validation loss improvement, allowing the optimizer to slide into cleaner minima.
- **Early Stopping:** Terminates training after 5 epochs of degrading validation performance and restores the best weights automatically.
- **LR History Plot:** Visualizes exactly which epochs the scheduler fired — useful for understanding training dynamics.
- **Diagnostic Reporting:** Classification report (Precision, Recall, F1), Seaborn confusion matrix heatmap, and a subplot of the hardest failure cases the model missed.

---

## 🛠️ Tech Stack

- **Runtime:** Python 3
- **Neural Network:** TensorFlow 2.x & Keras
- **UI:** Gradio
- **Image Processing:** Pillow (PIL)
- **Numerics & Evaluation:** NumPy & scikit-learn
- **Visualization:** Matplotlib & Seaborn

---

## 🚀 How to Run

### Google Colab (Recommended)

1. Upload `mnist_digit_recognizer.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all** (`Ctrl+F9`)
4. Scroll to the last cell, draw a digit in the Gradio sketchpad, and read the top 3 predictions

### Local Machine

```bash
pip install tensorflow gradio notebook seaborn scikit-learn matplotlib numpy pillow

jupyter notebook mnist_digit_recognizer.ipynb
```

---

## 👤 Author

**[Mohamed Ouledali]** — Engineering Student
