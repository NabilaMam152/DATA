import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




#fonction pour charger les donnees
def read_data(filename):

    with open(filename,'r') as csvfile:
        datareader = csv.reader(csvfile)
        metadata = next(datareader)
        traindata=[]
        for row in datareader:
            traindata.append(row)

    return (metadata, traindata)

#fonction pour deviser les donnes de dataset.csv
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    testset = list(dataset)
    i=0
    while len(trainSet) < trainSize:
        trainSet.append(testset.pop(i))
    return [trainSet, testset]

#fonction qui classifie les donnes selon la methode de classification naive bayesienne
def classify(data,test):

    total_size = data.shape[0]
    print("\n")
    print("training data size=",total_size)
    print("test data size=",test.shape[0])
 #initialisation des variables à 0
    countHeartDisease = 0
    countNoHeartDisease = 0
    probHeartDisease = 0
    probNoHeartDisease = 0
    print("\n")
    print("target    count    probability")

    for x in range(data.shape[0]):
        # calculer le nombres de patients avec maladies cardiaques
        if data[x,data.shape[1]-1] == '1':
            countHeartDisease +=1
        # calculer le nombres de patients sans maladies cardiaques
        if data[x,data.shape[1]-1] == '0':
            countNoHeartDisease +=1
#appliquer le theorem : naive de bayes
    probHeartDisease=countHeartDisease/total_size
    probNoHeartDisease= countNoHeartDisease / total_size

    print('1',"\t",countHeartDisease,"\t",probHeartDisease)
    print('0',"\t",countNoHeartDisease,"\t",probNoHeartDisease)

#initialiser les variables a 0
    prob0 =np.zeros((test.shape[1]-1))
    prob1 =np.zeros((test.shape[1]-1))
    accuracy=0
    print("\n")
    print("instance prediction  target")

    for t in range(test.shape[0]):
        for k in range (test.shape[1]-1):
            count1=count0=0
            for j in range (data.shape[0]):
                #how many times appeared with 0
                if test[t,k] == data[j,k] and data[j,data.shape[1]-1]=='0':
                    count0+=1
                #how many times appeared with 1
                if test[t,k]==data[j,k] and data[j,data.shape[1]-1]=='1':
                    count1+=1
            prob0[k]=count0/countNoHeartDisease
            prob1[k]=count1/countHeartDisease

        probno=probNoHeartDisease
        probyes=probHeartDisease
        for i in range(test.shape[1]-1):
            probno=probno*prob0[i]
            probyes=probyes*prob1[i]
        if probno>probyes:
            predict='0'
        else:
            predict='1'

        print(t+1,"\t",predict,"\t    ",test[t,test.shape[1]-1])
        if predict == test[t,test.shape[1]-1]:
            accuracy+=1
    final_accuracy=(accuracy/test.shape[0])*100
    print("accuracy",final_accuracy,"%")
    return

metadata,traindata= read_data("dataset.csv")
splitRatio=0.8
trainingset, testset=splitDataset(traindata, splitRatio)
training=np.array(trainingset)
print("\n The Training data set are:")
for x in trainingset:
    print(x)

testing=np.array(testset)
print("\n The Test data set are:")
for x in testing:
    print(x)
classify(training,testing)



#reading the database
data = pd.read_csv("dataset.csv")
sns.scatterplot(x='age', y='sex', data=data,
               hue='target')



# Adding Title to the Plot
plt.title("maladies cardiaques")


plt.show()
