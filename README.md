

# 🌳 Custom Decision Tree Library

A fully custom implementation of a Decision Tree classifier in Python, built from scratch using core machine learning principles. This project includes training, prediction, evaluation, feature importance, visualization, and preprocessing utilities.

---

## 🚀 Features

- Decision Tree implementation (no sklearn tree used)
- Supports:
  - Gini impurity
  - Entropy / Information Gain
- Class weighting (including "balanced")
- Configurable hyperparameters:
  - max_depth
  - min_samples_split
  - min_samples_leaf
  - max_features
- Probability predictions (predict_proba)
- Custom threshold tuning (tune_threshold)
- Model evaluation (accuracy, precision, recall, F1, confusion matrix)
- Feature importance computation
- Graphviz-based tree visualization
- Gini-based feature selection with heatmap

---

## ⚙️ Installation

### Clone the repository
```bash
git clone https://github.com/your-username/Decision-Tree-Builder.git
cd Decision-Tree-Builder
```

### Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn graphviz
```

---

## 🧠 Usage

### 1. Basic Training
```bash
from DT_Library import DecisionTree
import pandas as pd

df = pd.read_csv("data.csv")

X = df.drop(columns=["target"])
y = df["target"]

tree = DecisionTree(
    mode="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=2
)

tree.fit(X, y)
```

### 2. Prediction
```bash
predictions = tree.predict(X)
probabilities = tree.predict_proba(X)
```

### 3. Evaluation
```bash
tree.evaluate(X, y, dataset_name="Train Set")
```

### 4. Threshold Tuning
```bash
best_threshold, best_score = tree.tune_threshold(X_val, y_val, metric='f1')
```

### 5. Visualizing the Tree
```bash
tree.visualize_tree(filename="my_tree")
```

### 6. Feature Importance
```bash
importance = tree.get_feature_importance()
print(importance)
```

### 7. Feature Selection
```bash
from DT_Library import heatmap_gini_filter_clean

clean_df, report = heatmap_gini_filter_clean(
    df,
    target_col="target",
    threshold=0.01
)
```
---

## 📌 Notes

- Designed for binary classification
- Class 1 is treated as the positive class
- Threshold tuning allows precision/recall tradeoff
- Handles class imbalance via weighting
