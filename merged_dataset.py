import pandas as pd

# read the datasets 
df1 = pd.read_csv(r"C:\Users\duygu\Desktop\CaloriesBurntML - Machine Learning Project\calories.csv") 
df2 = pd.read_csv(r"C:\Users\duygu\Desktop\CaloriesBurntML - Machine Learning Project\exercise.csv") 

merge_data = pd.merge(df1, df2, how='outer') 
print(merge_data) 

# Birleştirilmiş veriyi CSV dosyası olarak kaydet
merge_data.to_csv(r"C:\Users\duygu\Desktop\CaloriesBurntML - Machine Learning Project\merged_dataset.csv", index=False)



