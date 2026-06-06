# MNIST Logistic Regression Classifier

A high-performance handwritten digit classifier trained on the MNIST dataset using Logistic Regression. Achieves **98% test accuracy** on 70,000+ images through principled ML practices — L2 regularization, feature normalization, batch processing, and stratified k-fold cross-validation.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **98%** |
| Dataset Size | 70,000 images (60K train / 10K test) |
| Model | Logistic Regression (One-vs-Rest) |
| Validation Strategy | Stratified K-Fold Cross-Validation |

---

## Techniques Used

- **Feature Normalization** — pixel values scaled to [0, 1] to stabilize gradient updates and improve convergence
- **L2 Regularization** — penalizes large weights to reduce overfitting on high-dimensional input (784 features)
- **Stratified K-Fold Cross-Validation** — preserves class distribution across folds, ensuring unbiased generalization estimates on held-out data
- **Batch Processing** — processes data in mini-batches for memory-efficient training on the full 70K dataset
- **Hyperparameter Tuning** — systematic search over regularization strength (C) and solver settings to optimize validation accuracy
- **Confusion Matrix Evaluation** — per-class breakdown of predictions to identify misclassification patterns across all 10 digits

---

## Project Structure

```
mnist-logistic-classifier/
│
├── mnist_classifier.py       # Main training and evaluation script
├── requirements.txt          # Dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Requirements

```
scikit-learn
numpy
pandas
matplotlib
```

### Run

```bash
python mnist_classifier.py
```

The script will:
1. Download the MNIST dataset automatically via scikit-learn / keras datasets
2. Preprocess and normalize pixel features
3. Train the Logistic Regression model with cross-validation
4. Evaluate on the held-out test set and print accuracy + confusion matrix

---

## Key Design Decisions

**Why Logistic Regression on MNIST?**
While CNNs are the standard for image classification, Logistic Regression on MNIST is a strong baseline that demonstrates the power of classical ML when paired with good preprocessing. It's fully interpretable, fast to train, and achieves near-SOTA accuracy for a linear model — making it ideal for benchmarking and understanding the limits of linear classifiers on image data.

**Why Stratified K-Fold?**
MNIST has balanced classes, but stratified splitting guarantees class proportions are preserved in every fold — critical for getting reliable, unbiased cross-validation scores rather than lucky splits.

**Why L2 over L1?**
With 784 input features (28×28 pixels), L2 regularization distributes weight penalties smoothly across all features, which works better than L1's sparse solution for dense image data.

---

## Skills Demonstrated

`Python` `scikit-learn` `Machine Learning` `Logistic Regression` `Feature Engineering` `Hyperparameter Tuning` `Cross-Validation` `Model Evaluation` `NumPy` `Matplotlib`

---

## Author

**Tamanna Subudhi**  
B.S. Computer Science (AI Concentration) — Purdue University  
[LinkedIn](https://linkedin.com/in/tamanna-subudhi-6792a026a)
