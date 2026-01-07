# **Image Classification using CNN with Transfer Learning (Cats vs Dogs)**

## **Project Overview**

This project focuses on building a robust binary image classification model to distinguish between cats and dogs using Convolutional Neural Networks (CNNs).
Given the limited dataset size, the project emphasizes correct modeling strategy, proper evaluation, and practical deep learning decision-making rather than brute-force model complexity.

An initial CNN trained from scratch showed unstable validation behavior. To address this, transfer learning with a pretrained MobileNetV2 backbone was adopted, resulting in significantly improved and stable performance.

---

## **Dataset Description**

- Task: Binary image classification (Dog = 0, Cat = 1)

- Training samples: 2,000 images

- Test samples: 400 images

- Image shape: 100 × 100 × 3 (RGB)

- Pixel values: Normalized to [0, 1]

- Class balance: Perfectly balanced (50% Cat, 50% Dog)

The dataset was provided in flattened numerical format and reshaped into image tensors before modeling.

## **Modeling Approach**

- Transfer Learning using MobileNetV2

- Pretrained on ImageNet

- Backbone frozen to prevent overfitting

- Custom classifier head added:

  - Global Average Pooling

  - Dense layer (ReLU)

  - Dropout (0.5)

  - Sigmoid output for binary classification
 
### **Data Augmentation**

To improve generalization, the following augmentations were applied during training:

- Random rotations

- Width & height shifts

- Zoom

- Horizontal flips

---

## **Training & Validation Strategy**

- Explicit train–validation split (90% / 10%) using stratified sampling

- Data augmentation applied only to training data

- Validation and test sets kept completely unseen during training

- Optimizer: Adam

- Loss function: Binary Cross-Entropy

---

## **Model Performance**

### **Test Set Results**

- Test Accuracy: ~91.75%

- Test Loss: ~0.21

### **Confusion Matrix**

The final model shows balanced class-wise performance with no prediction bias toward a single class.

| **Actual \ Predicted** | **Dog (0)** | **Cat (1)** |
| ---------------------- | ----------- | ----------- |
| **Dog (0)**            | 180         | 20          |
| **Cat (1)**            | 13          | 187         |


### **Training Behavior**

- Training and validation accuracy curves closely track each other

- No signs of overfitting or instability

- Smooth and stable convergence

---

## **Key Insights**

- Training CNNs from scratch is not suitable for small image datasets

- Transfer learning enables strong performance even with limited data

- Proper evaluation logic (sigmoid + thresholding) is critical

- Data augmentation significantly improves robustness

- Correct validation strategy prevents data leakage and misleading results

---

## **Limitations**

- Limited dataset size: Performance may drop on more diverse real-world images

- Frozen backbone: Fine-tuning higher layers could further improve accuracy

- Binary-only classification: Not directly extensible to multi-class problems

- Resolution constraint: 100×100 resizing may remove fine-grained details

- No deployment testing: Latency and memory usage were not evaluated

---

## **Future Improvements**

- Fine-tune upper layers of MobileNetV2

- Expand dataset diversity

- Extend to multi-class animal classification

- Add explainability methods (e.g., Grad-CAM)

- Evaluate model under production constraints

---

## **Tools & Libraries**

- Python

- NumPy

- Matplotlib / Seaborn

- scikit-learn

- TensorFlow / Keras

- MobileNetV2 (Transfer Learning)

## **How to Run**

1. Clone the repository:
```bash
git clone https://github.com/PranayDomal/Cat-Dog-CNN.git
```

2. Navigate to the folder:
```bash
cd Cat-Dog-CNN
```

3. Run the notebook:
```bash
jupyter notebook image_classification_Dog/Cat.ipynb
```

## **Author**

https://www.linkedin.com/in/pranay-domal-a641bb368/
