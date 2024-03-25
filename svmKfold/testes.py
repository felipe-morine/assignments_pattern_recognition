from svmKfold import SvmKfold


svm = SvmKfold()

kernel= ["rbf","sigmoid"]
c =     [0.1,0.5,1]
gamma = [0.1,1,10]
dataset = ["dsFullComMedias&Dummies.csv", "dsComMediasBinariosSemDummies.csv"]


for i in range(0, len(dataset)):
    for n in range(0, len(kernel)):
        for j in range(0, len(c)):
            for p in range(0, len(gamma)):
                svm.svm(kernel[n],c[j],gamma[p],dataset[i])