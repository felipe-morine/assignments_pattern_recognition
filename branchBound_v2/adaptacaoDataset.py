
import pandas as pd
from sklearn import preprocessing

#importa o dataset
breast_cancer_filename = 'Dataset Full Adaptado PCA.xlsx'
breast_cancer_dataset = pd.read_excel(breast_cancer_filename,names=['class', 'age', 'menopause','tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])


#acessa somente os valores a serem normalizados
x = breast_cancer_dataset[['age','tumor-size', 'inv-nodes', 'deg-malig']].values 


#normaliza pelo mínimo e máximo
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)


#atualiza o dataset com os valores normalizados
breast_cancer_dataset[['age','tumor-size', 'inv-nodes', 'deg-malig']]= df[[0,1,2,3]]


#desconsidera atributos categóricos não binários para evitar usar dummies
del breast_cancer_dataset['menopause']
del breast_cancer_dataset['breast-quad']


# transformar binarios em 0-1
breast_cancer_dataset['class'] = breast_cancer_dataset['class'].replace('no-recurrence-events', 0)
breast_cancer_dataset['class'] = breast_cancer_dataset['class'].replace('recurrence-events', 1)

breast_cancer_dataset['node-caps'] = breast_cancer_dataset['node-caps'].replace('no', 0)
breast_cancer_dataset['node-caps'] = breast_cancer_dataset['node-caps'].replace('yes', 1)

breast_cancer_dataset['breast'] = breast_cancer_dataset['breast'].replace('left', 0)
breast_cancer_dataset['breast'] = breast_cancer_dataset['breast'].replace('right', 1)

breast_cancer_dataset['irradiat'] = breast_cancer_dataset['irradiat'].replace('no', 0)
breast_cancer_dataset['irradiat'] = breast_cancer_dataset['irradiat'].replace('yes', 1)

# tratar missing values usando moda
breast_cancer_dataset['node-caps'] = breast_cancer_dataset['node-caps'].replace('?', breast_cancer_dataset['node-caps'].value_counts().idxmax())


output_filename = 'breast_cancer_final.csv'
breast_cancer_dataset.to_csv(output_filename, index=False)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(breast_cancer_dataset)


