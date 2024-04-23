import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import fetch_openml

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.head(10000)
y = y[:10000].astype(int)
print(X.shape, y.shape)
print(X.head())