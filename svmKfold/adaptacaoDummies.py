import pandas as pd

df = pd.read_excel("dsFullComMedias.xlsx")

# tratar missing values usando moda
df['node-caps'] = df['node-caps'].replace('?', df['node-caps'].value_counts().idxmax())
df['breast-quad'] = df['breast-quad'].replace('?', df['breast-quad'].value_counts().idxmax())

#transforma os atributos categóricos em dummies
dsFullComDummies = pd.get_dummies(df, columns=["menopause", "node-caps",
               "breast", "breast-quad", "irradiat"])

dsFullComDummies.to_csv('dsFullComMedias&Dummies.csv', index=False)