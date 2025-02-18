# ==============================================
# 📌 1. Gerekli Kütüphaneleri Yükleme
# ==============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error as mae

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

# ==============================================
# 📌 2. Veriyi Yükleme ve İnceleme
# ==============================================
df = pd.read_csv("merged_dataset.csv")

# Veri hakkında genel bilgi
print(df.info())   
print(df.head())   
print(df.describe())  

# ==============================================
# 📌 3. Kategorik Veriyi Sayısal Formata Çevirme
# ==============================================
# Gender sütunu 'male' -> 0, 'female' -> 1 olacak şekilde çevriliyor
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

# ==============================================
# 📌 4. Hedef Değişken ve Özellikleri Ayırma
# ==============================================
features = df.drop(['User_ID', 'Calories'], axis=1)  # Özellik sütunları
target = df['Calories'].values  # Hedef değişken

# ==============================================
# 📌 5. Veriyi Eğitim ve Test Olarak Bölme
# ==============================================
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

# ==============================================
# 📌 6. Veriyi Normalleştirme (StandardScaler)
# ==============================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ==============================================
# 📌 7. Modelleri Eğitme ve Karşılaştırma
# ==============================================
models = [
    LinearRegression(),
    XGBRegressor(),
    Lasso(),
    RandomForestRegressor(),
    Ridge()
]

print("\n🔹 MODELLERİN PERFORMANSI 🔹\n")
for model in models:
    model.fit(X_train, Y_train)  # Modeli eğit

    print(f'{model} :')

    # Eğitim hatası hesapla
    train_preds = model.predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))

    # Test hatası hesapla
    val_preds = model.predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print()

# ==============================================
# 📌 8. En İyi Modeli Optimize Etme
# ==============================================
print("\n🔹 OPTİMİZE EDİLMİŞ MODEL 🔹\n")
optimized_rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
optimized_rf.fit(X_train, Y_train)

val_preds = optimized_rf.predict(X_val)
print("Optimized RandomForestRegressor Validation Error:", mae(Y_val, val_preds))
