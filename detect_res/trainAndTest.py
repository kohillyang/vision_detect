# #coding=utf-8
# import numpy as np
# import cv2,os
# import getpass 
# import numpy as np
# import theano
# import lasagne
# from random import randint
# from lasagne import layers
# from lasagne.updates import nesterov_momentum
# from nolearn.lasagne import NeuralNet
# from nolearn.lasagne import visualize
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from nolearn.lasagne import TrainSplit
# from nolearn.lasagne.visualize import draw_to_notebook
# from nolearn.lasagne.visualize import plot_loss
# from nolearn.lasagne.visualize import plot_conv_weights
# from nolearn.lasagne.visualize import plot_conv_activity
# from nolearn.lasagne.visualize import plot_occlusion
# from nolearn.lasagne.visualize import plot_saliency
# from lasagne.nonlinearities import softmax
# from lasagne.layers import DenseLayer
# from lasagne.layers import InputLayer
# from lasagne.layers import DropoutLayer
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
# from lasagne.nonlinearities import softmax
# from lasagne.updates import adam
# from lasagne.layers import get_all_params
# # from sklearn.neighbors import KNeighborsClassifier
# # classfier =KNeighborsClassifier(n_neighbors=30)
# # from sklearn.neural_network import MLPClassifier
# # classfier = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
# #        beta_1=0.9, beta_2=0.999, early_stopping=False,
# #        epsilon=1e-08, hidden_layer_sizes=(25,10), learning_rate='constant',
# #        learning_rate_init=0.001, max_iter=2000, momentum=0.9,
# #        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
# #        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
# #        warm_start=False)
# layers0 = [
#     # layer dealing with the input data
#     (InputLayer, {'shape': (None,1,64, 32)}),

#     # first stage of our convolutional layers
#     (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
#     # (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
#     # (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
#     # (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
#     # (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
#     # (MaxPool2DLayer, {'pool_size': 2}),

#     # # second stage of our convolutional layers
#     # (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
#     # (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
#     # (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
#     # (MaxPool2DLayer, {'pool_size': 2}),

#     # # two dense layers with dropout
#     # (DenseLayer, {'num_units': 64}),
#     # (DropoutLayer, {}),
#     (DenseLayer, {'num_units': 64}),

#     # the output layer
#     (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
# ]
# classfier = None
# x_test = []
# y_test = []
# x_train = []
# y_train = []
# def getVectorByFile(filename):
#     try:
#         img = cv2.imread(filename)
#         img = cv2.resize(img,(64, 32), interpolation = cv2.INTER_CUBIC)
#         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         r_x = gray.reshape(1,64,32).astype(np.float32)
#         r_x /= 255
#         r_x -= 0.5
#         return r_x
#     except Exception as e:
#         print("[error]:Read img failed",e)
#         return None
# def init():
#     import pickle
#     classfier_pickle_path = "/home/"+getpass.getuser()+"/kohillyang/detect_res/classfier.pickle"
#     if os.path.exists(classfier_pickle_path):
#         with open(classfier_pickle_path,"rb") as f:
#             global classfier
#             classfier = pickle.load(f) 
#     else:
#         global classfier
#         classfier = NeuralNet(
#             layers=layers0,

#             # optimization method params
#             update=nesterov_momentum,
#             update_learning_rate=1,
#             update_momentum=0.9,
#             max_epochs=10,
#             verbose=1,   
#             objective_l2=0.0025,

#             train_split=TrainSplit(eval_size=0.25),    
#                     )        
#         pos_lable = np.int32(1)
#         neg_lable = np.int32(0)
#         dataSetPath = "/home/"+getpass.getuser()+"/kohillyang/detect_res/cl"
#         for x,y,names in os.walk(dataSetPath +"/0/"):
#             for name in names:
#                 x_train.append(getVectorByFile(x+name))
#                 y_train.append(neg_lable)
#         for x,y,names in os.walk(dataSetPath +"/1/"):
#             for name in names:
#                 x_train.append(getVectorByFile(x+name))
#                 y_train.append(pos_lable)      


#         for x,y,names in os.walk(dataSetPath +"/test/1/"):
#             for name in names:
#                 x_test.append(getVectorByFile(x+name))
#                 y_test.append(pos_lable)

#         for x,y,names in os.walk(dataSetPath +"/test/0/"):
#             for name in names:
#                 x_test.append(getVectorByFile(x+name))
#                 y_test.append(neg_lable)
#         x_train_last = []
#         y_train_last = []
#         while True:
#             index = randint(0,len(x_train)-1)
#             x_train_last.append(x_train[index])
#             y_train_last.append(y_train[index])
#             del x_train[index]
#             del y_train[index]
#             if len(x_train) is 0:
#                 break        
#         print(np.array(x_train_last).shape)
#         print(np.array(y_train_last).shape)
#         print(y_train_last[0])
#         classfier.fit(np.array(x_train_last),np.array(y_train_last))
#         prediction = lasagne.layers.get_output(classfier)
#         loss = lasagne.objectives.categorical_crossentropy(prediction, y_train_last)
#         loss = loss.mean()
#         result = classfier.predict(x_test)
#         matches = result==y_test
#         print(result)
#         print(y_test)
#         correct = np.count_nonzero(matches)
#         accuracy = correct*100.0/len(result)
#         print (accuracy)

# def detect(img):
#     r = classfier.predict([img])
#     return int(r)
# if __name__ =="__main__":
#     init()

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import sys,os
classfier =KNeighborsClassifier(n_neighbors=30)
def getVectorByFile(filename):
    try:
        img = cv2.imread(filename)
        img = cv2.resize(img,(64, 32), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        r_x = gray.reshape(64*32,).astype(np.float32)
        r_x /= 255
        r_x -= 0.5
        return r_x
    except Exception as e:
        print("[error]:Read img failed",e)
        return None
def init():
    rootpath = './cl/'
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
def detect(da):
    return True
if __name__=="__main__":
    init()