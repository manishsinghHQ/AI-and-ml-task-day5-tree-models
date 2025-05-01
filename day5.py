from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
dataset=pd.read_csv("Titanic-Dataset.csv")
a=dataset.info()
print(a)