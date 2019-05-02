'''
Exercise 2
by Biagio Brattoli
'''
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from scipy.optimize import leastsq
from sklearn.svm import SVC, LinearSVC

# function to load images and labels of the original CIFAR10 dataset
from cifar10 import load_CIFAR10


cifar10_dir = './cifar-10-batches-py'# download the files from the official website

######### Provided Functions ############
def get_features(img,bins=10):
    edges = np.linspace(0,255,bins+1)
    red,_ = np.histogram(img[:,:,0].flatten(),edges)
    green,_ = np.histogram(img[:,:,1].flatten(),edges)
    blue,_ = np.histogram(img[:,:,2].flatten(),edges)
    return np.concatenate([red,blue,green]).reshape([1,-1])

def get_data():
    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)
    #compute the features
    X_train_feat = np.concatenate([get_features(img,10) for img in tqdm(X_train)])
    X_test_feat  = np.concatenate([get_features(img,10) for img in tqdm(X_test)])
    return X_train_feat, Y_train, X_test_feat, Y_test

def get_two_classes(X_train_feat, Y_train, X_test_feat, Y_test):
    selection = np.logical_or(Y_train==0,Y_train==6) #Airplane VS Frog
    x, y = X_train_feat[selection], Y_train[selection]
    y[y==np.unique(y)[0]] = -1
    y[y==np.unique(y)[1]] =  1
    
    selection = np.logical_or(Y_test==0,Y_test==6)
    x_test, y_test = X_test_feat[selection], Y_test[selection]
    y_test[y_test==np.unique(y_test)[0]] = -1
    y_test[y_test==np.unique(y_test)[1]] =  1
    
    return x,y,x_test,y_test

def get_multiple_classes(X_train_feat, Y_train, X_test_feat, Y_test):
    selection = np.logical_or(Y_train==0,Y_train==6) #Airplane and Frog
    selection = np.logical_or(selection,Y_train==5)  # add Dogs
    selection = np.logical_or(selection,Y_train==9)  # add Trucks
    
    x, y = X_train_feat[selection], Y_train[selection]
    y[y==0] = 0
    y[y==5] = 1
    y[y==6] = 2
    y[y==9] = 3
    
    selection = np.logical_or(Y_test==0,Y_test==6) #Airplane and Frog
    selection = np.logical_or(selection,Y_test==5) # add Dogs
    selection = np.logical_or(selection,Y_test==9) # add Trucks
    
    x_test, y_test = X_test_feat[selection], Y_test[selection]
    y_test[y_test==0] = 0
    y_test[y_test==5] = 1
    y_test[y_test==6] = 2
    y_test[y_test==9] = 3
    
    return x,y,x_test,y_test

######### Solutions ############
#Load Data globally
X_train_feat, Y_train, X_test_feat, Y_test = get_data()

# Implement a linear classifier solved using least-square.
# Suggestion: use the function "leastsq"
class LSQclassifier:
    weigths = np.array(0)
    bias = 0
    def __init__(self):
        pass
    
    def fit(self,x,y):
        #TODO
        for i in range len(y):
            ErrorFunc = y - (lambda weigths * x + bias)
            leastsq(ErrorFunc, weights, bias, x, y)
        return self
    
    def predict(self,x):
        #TODO
        pred = weights * x + bias
        return pred

# Solve binary task using a linear classifier trained by least-square
def task1():
    # Create binary task
    x,y,x_test,y_test = get_two_classes(X_train_feat, Y_train, X_test_feat, Y_test)
    # Train Linear classifier for binary classification using LSQclassifier
    #TODO
    pipeline    = LSQclassifier.fit(x, y)
    pred_train  = pipeline.predict(x)
    pred_test   = pipeline.predict(x_train)
    train_acc   = accuracy_score(y,pred_train)
    test_acc    = accuracy_score(y_test,pred_test)
    print('Linear classifier by Least-Square: Train %.2f, Test %.2f'%(train_acc,test_acc))
    # Plot learned weights using plt.bars
    # TODO
    plt.bars(pipeline.weigths)

# Binary classification using Linear SVM
def task2():
    # Create binary task
    x,y,x_test,y_test = get_two_classes(X_train_feat, Y_train, X_test_feat, Y_test)
    # Train Linear SVM classifier for binary classification
    # Use LinearSVC()
    #TODO
    lsvc = LinearSVC()
    lscv.fit(x, y)
    pred_train = lscv.predict(x)
    pred_test  = lscv.predict(x_test)
    train_acc  = accuracy_score(y,pred_train)
    test_acc   = accuracy_score(y_test,pred_test)
    print('Linear SVM classifier: Train %.2f, Test %.2f'%(train_acc,test_acc))
    # Plot learned weights using plt.bars
    # TODO
    plt.bars(lscv.get_params())


# Compare in words the result and weights from least-square and svm
#TODO


# Binary classification using Non-linear SVM
def task3():
    # Create binary task
    x,y,x_test,y_test = get_two_classes(X_train_feat, Y_train, X_test_feat, Y_test)
    # Find best hyper-parameters (soft-margin C, kernels and gamma) using cross-validation
    #TODO
    
    # Train Linear SVM classifier for binary classification
    #TODO
    
    train_acc = accuracy_score(y,pred_train)
    test_acc  = accuracy_score(y_test,pred_test)
    print('Non-Linear SVM classifier: Train %.2f, Test %.2f'%(train_acc,test_acc))

# Comment the result obtained using non-linear svm
#TODO


# Solve multi-class task using a linear classifier trained by least-square
def task4():
    # Create binary task
    x,y,x_test,y_test = get_multiple_classes(X_train_feat, Y_train, X_test_feat, Y_test)
    # Implement a multi-class classifier using binary LSQclassifier in one-vs-all setup
    # TODO
    
    train_acc = accuracy_score(y,pred_train)
    test_acc  = accuracy_score(y_test,pred_test)
    print('Multi-class) Linear classifier by Least-Square: Train %.2f, Test %.2f'%(train_acc,test_acc))


## Solve multi-class task using a linear svm
def task5():
    # Create binary task
    x,y,x_test,y_test = get_multiple_classes(X_train_feat, Y_train, X_test_feat, Y_test)
    # Train multi-class linear svm
    #TODO
    
    train_acc = accuracy_score(y,pred_train)
    test_acc  = accuracy_score(y_test,pred_test)
    print('Multi-class) Linear SVM: Train %.2f, Test %.2f'%(train_acc,test_acc))



# Train linear logistic regression using Pytorch
# Answer questions from exercise pdf
def task6():
    ######### Linear Classifier using Pytorch classifier ############
    import torch
    import torch.nn as nn
    class LogisticRegression(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, num_classes)
        
        def forward(self, x):
            out = self.linear(x)
            return out
    
    # Create binary task
    x,y,x_test,y_test = get_multiple_classes(X_train_feat, Y_train, X_test_feat, Y_test)
    
    model = LogisticRegression(x.shape[1], len(np.unique(y)))
    
    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    
    X = torch.from_numpy(x.astype('float32'))
    Y = torch.from_numpy(y)
    
    num_epochs = 2000
    for epoch in range(num_epochs):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if epoch%1000==0 and epoch>0:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        if epoch%100==0:
            print ('Epoch: [%d/%d], Loss: %.4f'%(
                epoch+1, num_epochs,loss.item()))
    
    pred = model(torch.from_numpy(x.astype('float32'))).argmax(1)
    train_acc  = accuracy_score(y,pred.detach().numpy())
    pred = model(torch.from_numpy(x_test.astype('float32'))).argmax(1)
    test_acc  = accuracy_score(y_test,pred.detach().numpy())
    print('Linear classifier by least square: Train %.2f, Test %.2f'%(train_acc,test_acc))
    
    #Plot the loss over time
    #TODO


if __name__ == '__main__':
  task1()
  task2()
  task3()
  task4()
  task5()
  task6()
  
