import pandas as pd

df_coim = pd.read_csv("C:\\Users\\jbpra\\Coding\\Uyir_Project\\dataset\\coimbatore.csv")

df_acc = pd.read_csv("C:\\Users\\jbpra\\Coding\\Uyir_Project\\dataset\\dataset_traffic_accident_prediction1.csv")

df_final = pd.merge(df_coim, df_acc, on='Weather', how='left')

df_final = df_final.drop('Accident_Held', axis=1)

df_final.to_csv('merged.csv', index=False)