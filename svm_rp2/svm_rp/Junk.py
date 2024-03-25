import pandas as pd
from sklearn.preprocessing import StandardScaler
import kfold


breast_cancer_dataset = pd.read_csv('dsFullComMedias&Dummies.csv')

# breast_cancer_dataset['class'] = breast_cancer_dataset['class'].replace('no-recurrence-events', 0)
# breast_cancer_dataset['class'] = breast_cancer_dataset['class'].replace('recurrence-events', 1)

# breast_cancer_dataset.to_csv('dsFullComMedias&Dummies.csv', index=False)

folds_list = kfold.k_fold(breast_cancer_dataset, 3, 'class')

# scaler = StandardScaler.fit()
print(len(folds_list))