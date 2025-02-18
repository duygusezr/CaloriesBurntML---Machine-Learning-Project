
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

# Veri setinin genel bilgilerini göster
print(df.info())  # Sütun isimleri, veri tipleri ve eksik değerleri görüntüle

print(df.shape)   #Veri setinin satır ve sütun sayısını gösterir.

# Veri setindeki sayısal değerlerin istatistiklerini göster
print(df.describe())  # Ortalama, min, max, standart sapma gibi istatistikleri görüntüle

df.replace({'male': 0, 'female': 1},inplace=True)
print(df.head())

plt.figure(figsize=(8, 8))
sb.heatmap(df.corr() > 0.9,annot=True,cbar=False)
plt.show()






