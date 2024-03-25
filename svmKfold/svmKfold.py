from decimal import getcontext, Decimal
from sklearn.svm import SVC  
from sklearn.metrics import  precision_recall_fscore_support, accuracy_score
from training_test import TrainnigTestFolds

class SvmKfold:
    
    def __init__(self):       
        #inicializa totais das estimativas
        self.totNoRecPrecision =0
        self.totNoRecRecall = 0
        self.totAccuracy = 0   
        self.dataset = ""
        self.kernel = ""
        self.gamma = 0
        self.c = 0
        #kpartes
        self.kPartes = 17
    
    def svm(self,kernel, c, gamma, dataset):
        #armazena para pode acessar em outro método
        self.dataset = dataset
        self.kernel = kernel
        self.gamma = gamma
        self.c = c
        self.totNoRecPrecision =0
        self.totNoRecRecall = 0
        self.totAccuracy = 0
        
        #informa o dataset a ser utilizado
        ttf = TrainnigTestFolds(dataset)
        
        #define os parâmetros do svm
        svclassifier = SVC(kernel=self.kernel, C=self.c, gamma = self.gamma)

        
        #para cada conjunto de treinamento e teste dados pelo k-fold
        for i in range(0, self.kPartes -1):
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
            self.totNoRecPrecision += precision_recall_fscore_support(y_test,y_pred)[0][0]
            self.totNoRecRecall += precision_recall_fscore_support(y_test,y_pred)[0][1]
            self.totAccuracy += accuracy_score(y_test, y_pred)
            
        self.geraMedias()
                
    def geraMedias(self):

        #gera a media entre as k-partes
        mediaNoRecPrecision = round(self.totNoRecPrecision/self.kPartes,3)
        mediaNoRecRecall = round(self.totNoRecRecall/self.kPartes,3)
        mediaNoRecAccurary = round(self.totAccuracy/self.kPartes,3)
        
        #imprime o cabeçalho
        print("################################################################################")
        print("Dataset: " + self.dataset + " # Kernel: " + self.kernel + " # Gamma: " + str(self.gamma) + " # C: " + str(self.c))
        print("--------------------------------------------------------------------------------")
        #imprime as medidas
        print("Precisão: " + str(mediaNoRecPrecision)+" - Recall: " + str(mediaNoRecRecall)+ " - Acurácia: " + str(mediaNoRecAccurary))








