import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv , os

# gradient decent
def GD(X,Y,W,Iteration,eta):
    listCost = []
    for itera in range(Iteration):
        arrayY1 = np.dot(X,W)
        arrayLoss = arrayY1 - Y
        arrayCost = (np.sum(arrayLoss**2)/X.shape[0])
        listCost.append(arrayCost)

        #Gradient
        arrayGradient = (X.T.dot(arrayLoss) / X.shape[0])
        
        W -= eta*arrayGradient

        if itera % 1000 == 0:
            print("iteration:{}, cost:{} ".format(itera, arrayCost))
    return W,listCost
#Training數據擷取
def traindataProcess():

    #Declare a 18-dim vector (Data)
    listTrainData = []
    for i in range(18):
        listTrainData.append([])
    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    fn = "data/train.csv"
    dirpath = os.path.join(BASEDIR, fn)
    
    rowTrain = pd.read_csv(dirpath,usecols=range(3,27),encoding="big5")
    #替換字元
    rowTrain = rowTrain.replace(['NR'],[0.0])
    #把int轉乘float
    rowTrain = np.array(rowTrain).astype(float)
    
    n_row = 0
    for r in rowTrain:  
        for i in range(0,24):
            listTrainData[ n_row % 18].append(r[i])
        n_row += 1

    listTrain_X = []
    listTrain_Y = []
    for m in range(12):
        #一個月每9小時算一筆資料，第10比資料當label
        for i in range(471):            
            #the value of 10th-hr pm2.5
            listTrain_Y.append(listTrainData[9][480*m + i + 9])
            #previous 9-hr data
            listTrain_X.append([])
            for p in range(18):
                for t in range(9):
                    listTrain_X[471*m+i].append(listTrainData[p][480*m+i+t])
                
    arrayTrainX = np.array(listTrain_X)  #shape(5652,162)
    arrayTrainY = np.array(listTrain_Y)  #shape(5652,)


        # 增加bias項 shape(5652,163)
    arrayTrainX = np.concatenate((np.ones((arrayTrainX.shape[0], 1)), arrayTrainX), axis=1) # (5652, 163)
    return arrayTrainX , arrayTrainY
#Test數據擷取
def testdataProcess():
    '''
    tn = "test.csv"
    rowTest = pd.read_csv(tn,usecols=range(2,11),encoding="big5")
    rowTest = rowTest.replace(['NR'],[0.0])
    rowTest = np.array(rowTest).astype(float)
    
    listTestData = []
    n_row=0
    for r in rowTest:         
        if n_row % 18 == 0:
            listTestData.append([])
            for i in range(0,9):
                listTestData[n_row//18].append(r[i])
        else:
            for i in range(0,9):
                listTestData[n_row//18].append(r[i])
        n_row += 1
    
    arrayTestX = np.array(listTestData)
    return arrayTestX
    '''
    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    fn = "data/test.csv"
    dirpath = os.path.join(BASEDIR, fn)
    
    listTestData = []
    textTest = open( dirpath, "r", encoding="big5")
    rowTest = csv.reader(textTest)
    n_row = 0
    for r in rowTest:
        if n_row % 18 == 0:
            listTestData.append([])
            for i in range(2, 11):
                listTestData[n_row // 18].append(float(r[i]))
        else:
            for i in range(2, 11):
                if r[i] == "NR":
                    listTestData[n_row // 18].append(float(0))
                else:
                    listTestData[n_row // 18].append(float(r[i]))
        n_row += 1
    textTest.close()
    arrayTestX = np.array(listTestData) #TestX =  (240, 162)
    return arrayTestX

def main():
    #Data資料輸入出
    arrayTrainX , arrayTrainY = traindataProcess()
    arrayTestX = testdataProcess()

    #gradient decent
    # x1,x2,x3...,xn = 1xp
    
    arrayW = np.zeros(arrayTrainX.shape[1])
    intLearningRate = 1e-6
    arrayW_gd, listCost_gd = GD(X=arrayTrainX , Y=arrayTrainY , W=arrayW , Iteration=20000 ,eta=intLearningRate)

    #Test
    arrayTestX = np.concatenate((np.ones((arrayTestX.shape[0], 1)), arrayTestX), axis=1)
    arrayPredictY_gd = np.dot(arrayTestX,arrayW_gd)

    #visualization
    plt.plot(np.arange(len(listCost_gd[3:])), listCost_gd[3:], "b--", label="GD_0")
    plt.title("Train Process")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function(MSE)")
    plt.legend()
    plt.savefig("TrainProcess")
    plt.show()

    # visualize predict value with different methods
    plt.figure(figsize=(12, 4))
 
    plt.plot(np.arange(len(arrayPredictY_gd)), arrayPredictY_gd, "g--")
    plt.title("GD")
    plt.xlabel("Test Data Index")
    plt.ylabel("Predict Result")
    plt.tight_layout()
    plt.savefig("Compare")
    plt.show()

if __name__ == '__main__':
    main()
    

    



