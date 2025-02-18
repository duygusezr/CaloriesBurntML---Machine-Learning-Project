
# ==============================================
# 📌 Importing Libraries and Dataset
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# CSV dosyasını yükle
df = pd.read_csv('merged_dataset.csv')

sb.scatterplot(x='Height', y='Weight', data=df) 
plt.show()

features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x)
plt.tight_layout()
plt.show()

features = df.select_dtypes(include='float').columns

features = df.select_dtypes(include='float').columns

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot((len(features) // 3) + 1, 3, i + 1)  # Dinamik grid ayarlaması
    sb.histplot(df[col], kde=True)  # distplot yerine histplot kullandık
plt.tight_layout()
plt.show()





