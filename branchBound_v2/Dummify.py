import pandas as pd

breast_cancer_filename = 'breast_cancer_original.csv'
breast_cancer_dataset = pd.read_csv(breast_cancer_filename)



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
breast_cancer_dataset['breast-quad'] = breast_cancer_dataset['breast-quad'].replace('?', breast_cancer_dataset['breast-quad'].value_counts().idxmax())

# colocar dummies
breast_cancer_dataset = pd.get_dummies(breast_cancer_dataset)

output_filename = 'breast_cancer_final.csv'
breast_cancer_dataset.to_csv(output_filename, index=False)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(breast_cancer_dataset)
