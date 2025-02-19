# 📊 CaloriesBurntML - Machine Learning Project  

Bu proje, farklı fiziksel özellikler ve aktivite verilerine dayanarak **kalori yakımını tahmin etmek** için bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. 🚀  

---

## 📌 Proje Açıklaması  
Bu proje, kullanıcıların yaş, boy, kilo, egzersiz süresi, kalp atış hızı ve vücut sıcaklığı gibi verilerini kullanarak **yakılan kalori miktarını tahmin eden bir model** oluşturur. Çeşitli **makine öğrenmesi algoritmaları** eğitilmiş ve karşılaştırılmıştır.

📌 **Kullanılan Modeller:**  
- **Linear Regression**  
- **XGB Regressor**  
- **Lasso Regression**  
- **Random Forest Regressor**  
- **Ridge Regression**  

En iyi modeli belirlemek için `mean_absolute_error (MAE)` metriği kullanılmıştır.

---

## 📂 Veri Seti  
Proje kapsamında iki veri seti kullanılmış ve birleştirilmiştir:  

1. **calories.csv** → Kullanıcıların **yakılan kalori** miktarını içeren veri seti.  
2. **exercise.csv** → Kullanıcıların **yaş, boy, kilo, egzersiz süresi, kalp atış hızı ve vücut sıcaklığı** gibi özelliklerini içeren veri seti.  
3. **merged_dataset.csv** → Yukarıdaki iki veri setinin birleştirilmiş hali.  

📝 **Veri Kümesi Sütunları:**  
| Sütun Adı  | Açıklama |
|------------|------------------------------------------------|
| `User_ID`  | Kullanıcı kimliği |
| `Calories` | Yakılan kalori miktarı |
| `Gender`   | Cinsiyet (0: Erkek, 1: Kadın) |
| `Age`      | Yaş |
| `Height`   | Boy (cm) |
| `Weight`   | Kilo (kg) |
| `Duration` | Egzersiz süresi (dk) |
| `Heart_Rate` | Kalp atış hızı (BPM) |
| `Body_Temp` | Vücut sıcaklığı (°C) |

---

## 🛠️ Kullanılan Kütüphaneler  
Bu projede aşağıdaki Python kütüphaneleri kullanılmıştır:  

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
