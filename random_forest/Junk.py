import pandas as pd

df = pd.read_csv('breast_cancer_original.csv')
df.drop('breast-quad', inplace=True, axis=1)

df.to_csv('breast_cancer_mod.csv', index=False)