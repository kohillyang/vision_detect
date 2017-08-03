# #coding=utf-8

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import cv2
import sys,os
# classfier =KNeighborsClassifier(n_neighbors=30)
# classfier = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
classfier = MLPClassifier(
    hidden_layer_sizes=(100,), 
    activation='relu', 
    solver='adam', 
    alpha=0.0001, 
    batch_size='auto', 
    learning_rate='constant', 
    learning_rate_init=0.001, 
    power_t=0.5, 
    max_iter=10, 
    shuffle=True, 
    random_state=None, 
    tol=0.0001, 
    verbose=True, 
    warm_start=False, 
    momentum=0.9, nesterovs_momentum=True, 
    early_stopping=False, 
    validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
def getVectorByFile(filename):
    try:
        img = cv2.imread(filename)
        img = cv2.resize(img,(32, 64), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        r_x = gray.reshape(64*32,).astype(np.float32)
        r_x /= 255
        r_x -= 0.5
        return r_x
    except Exception as e:
        print("[error]:Read img failed",e)
        return None
def init():
    try:
        rootpath = os.path.dirname(os.path.realpath(__file__)) + "/cl/"
        print '[info]:rootpath,', rootpath
        posPath = rootpath  + "./1/"
        negPath = rootpath  + "./0/"
        global classfier
        x_train = []
        y_train = []
        for label in [0,1]:
            for n,p,q in os.walk(rootpath+str(label)+"/"):
                for name in q:
                    fp = os.path.join(n,name)
                    x_train.append(getVectorByFile(fp))
                    y_train.append(label)      
        classfier.fit(np.array(x_train),np.array(y_train))          
     
        x_test = []
        y_test = []
        for label in [0,1]:
            for n,p,q in os.walk(rootpath+"/test/"+str(label)+"/"):
                for name in q:
                    fp = os.path.join(n,name)
                    x_test.append(getVectorByFile(fp))
                    y_test.append(label)      
        result = classfier.predict(x_test)
        ra = result == y_test
        print "ratio",1.0* np.sum(ra) / len(ra), len(ra)
    except Exception as e:
        print e 
    pass
count = 0
def detect(da):
    try:
        global count
        count += 1
        x = np.array(da).astype(np.float32)
        x /= 255
        x -= 0.5
        assert(len(x) == 64 * 32)
        [y] = classfier.predict([x])
        x = x.reshape((32,64))
        x += 0.5
        x *= 255
        cv2.imwrite(filename,x)
        print "INFO,Detetc result:",y
        return y
    except Exception as e:
        print e
        return 0
if __name__=="__main__":
    init()