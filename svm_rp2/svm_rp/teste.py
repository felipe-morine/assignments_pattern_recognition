from sklearn.svm import SVC  
from sklearn.metrics import  precision_recall_fscore_support, accuracy_score
from training_test import TrainnigTestFolds
import pandas as pd

#inicializa totais das estimativas
totNoRecPrecision =0
totNoRecRecall = 0
totAccuracy = 0

#kpartes
kPartes = 17

#informa o dataset a ser utilizado via construtor
ttf = TrainnigTestFolds("dataset_dummies.csv")
# ttf = TrainnigTestFolds("dsComMediasBinariosSemDummies.csv")


#escolha do kernel
#svclassifier = SVC(kernel='linear')#linear(apenas para comparacao)
#svclassifier = SVC(kernel='poly', degree=7)#Polynomial
svclassifier = SVC(kernel='poly', C=0.5, gamma=1 )#Gaussian
#svclassifier = SVC(kernel='sigmoid')#Sigmoid


#para cada conjunto de treinamento e teste dados pelo k-fold
for i in range(0, kPartes -1):
    df_training = ttf.k_df_training(i) 
    df_test = ttf.k_df_test(i)
    
    #separa os atributos e as classes de teste e treinamento
    X_train = df_training.drop('class', axis=1)  
    y_train = df_training['class']
    X_test = df_test.drop('class', axis=1)
    y_test = df_test['class']
    
    #treina
    svclassifier.fit(X_train, y_train) 
    
    #estimacao dos testes
    y_pred = svclassifier.predict(X_test)  
        
    #pega a precisao e recall da classe prevista como noRecorrenceEvents
    #se quiser mudar para recorrenceEventes, trocar os indices dos registros de 0 para 1

    print(y_test, y_pred)

    totNoRecPrecision += precision_recall_fscore_support(y_test,y_pred)[0][0]
    totNoRecRecall += precision_recall_fscore_support(y_test,y_pred)[0][1]
    totAccuracy += accuracy_score(y_test, y_pred)
        
#gera a media entre as k-partes
mediaNoRecPrecision = totNoRecPrecision/kPartes
mediaNoRecRecall = totNoRecRecall/kPartes
mediaNoRecAccurary = totAccuracy/kPartes

#imprime as medidas
print("Precisão  : " + str(mediaNoRecPrecision))
print("Recall    : " + str(mediaNoRecRecall))
print("Acurácia  : " + str(mediaNoRecAccurary))







