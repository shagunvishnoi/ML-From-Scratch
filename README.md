# Machine Learning From Scratch

Implemented core machine learning algorithms from first principles using Python and NumPy, with Scikit-Learn used only for benchmarking.

## Algorithms Implemented

- Linear Regression
- Logistic Regression
- Principal Component Analysis (PCA)
- K-Means Clustering
- Decision Tree
- Random Forest

## Results

| Algorithm | My Implementation | Scikit-Learn |
|---|---:|---:|
| Linear Regression | MSE: 78.05446 | MSE: 78.05425 |
| Logistic Regression | Accuracy: 94.74% | Accuracy: 95.61% |
| Decision Tree | Accuracy: 92.98% | Accuracy: 94.74% |
| Random Forest | Accuracy: 95.61% | Accuracy: 96.49% |

## Visualizations

### Linear Regression Fit
![Linear Regression](images/linear_regression_fit.png)

### Gradient Descent Loss Curve
![Loss Curve](images/loss_curve.png)

### Logistic Regression Loss Curve
![Logistic Loss](images/logistic_loss_curve.png)

### PCA From Scratch
![PCA](images/pca_from_scratch.png)

### K-Means Clustering
![KMeans](images/kmeans_clusters.png)

### Decision Tree Visualization
![Decision Tree](images/decision_tree_visualization.png)

## Concepts Covered

- Gradient Descent
- Mean Squared Error
- Binary Cross-Entropy
- Sigmoid Function
- Entropy and Information Gain
- Bootstrap Aggregation
- Eigenvalues and Eigenvectors
- Covariance Matrix
- Euclidean Distance
- Centroid Optimization

## Project Structure

```text
ML-From-Scratch/
├── images/
├── models/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── pca.py
│   ├── kmeans.py
│   ├── decision_tree.py
│   └── random_forest.py
├── notebooks/
│   ├── linear_demo.py
│   ├── logistic_demo.py
│   ├── pca_demo.py
│   ├── kmeans_demo.py
│   ├── decision_tree_demo.py
│   └── random_forest_demo.py
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
```

Run any demo:

```bash
python notebooks/linear_demo.py
python notebooks/logistic_demo.py
python notebooks/pca_demo.py
python notebooks/kmeans_demo.py
python notebooks/decision_tree_demo.py
python notebooks/random_forest_demo.py
```

## Why This Project

Most machine learning projects use high-level libraries directly. This repository focuses on implementing algorithms from scratch to understand the mathematical foundations behind model training, optimization, dimensionality reduction, clustering, and ensemble learning.

## Future Work

- Naive Bayes
- K-Nearest Neighbors
- Support Vector Machine
- Neural Network with Backpropagation