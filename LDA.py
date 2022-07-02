# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 11:29:33 2022

@author: Seshu Kumar Damarla
"""

"""
Linear Discriminant Analysis for multi-class classification problem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc

gc.collect()

data = pd.read_csv('data.csv',header=None)

xydata = np.array(data)
"""" test observation """
test_x=np.array([[6,130,8]])   

xdata=xydata[:,1:]
N=xydata.shape[0]    # no. of examples

# noralization
xmax = np.amax(xdata, axis = 0, keepdims=True)
xmin = np.amin(xdata, axis = 0, keepdims=True)

xdata = (xdata - xmin) / (xmax-xmin)
print(xdata)

dd=xydata[:,0] == 1
c1xdata=xdata[dd==True,:]
c2xdata=xdata[dd==False,:]

C=np.unique(xydata[:,0])
C=C.astype(int)
C=np.array([C])

mean_class1 = np.mean(c1xdata, axis=0, keepdims=True, dtype=np.float)
mean_class2 = np.mean(c2xdata, axis=0, keepdims=True, dtype=np.float)

#print(mean_class1)
#print(np.transpose(mean_class1))

cov_class1 = np.cov(c1xdata,rowvar=False)
cov_class2 = np.cov(c2xdata,rowvar=False)

S=(1/(N-C.shape[1]))*(cov_class1 + cov_class2)
#print(S)
#print(np.linalg.inv(S))

prior=np.zeros([1,C.shape[1]])
prior[:,0] = c1xdata.shape[0] / N
prior[:,1] = c2xdata.shape[0] / N

#print(np.log(prior[:,0]))
# discriminate function

#delta_class1 = test_x * (np.linalg.inv(S)) * np.transpose(mean_class1) -(1/2) *(mean_class1) * (np.linalg.inv(S)) * np.transpose(mean_class1) + np.log(prior[:,0])
#delta_class2 = test_x * (np.linalg.inv(S)) * np.transpose(mean_class2) -(1/2) *(mean_class2) * (np.linalg.inv(S)) * np.transpose(mean_class2) + np.log(prior[:,1])

test_x = (test_x -xmean) / xstd

delta_class1 = np.dot(np.dot(test_x,np.linalg.inv(S)),np.transpose(mean_class1)) - (1/2) * np.dot(np.dot(mean_class1,np.linalg.inv(S)),np.transpose(mean_class1)) + np.log(prior[:,0])
delta_class2 = np.dot(np.dot(test_x,np.linalg.inv(S)),np.transpose(mean_class2)) - (1/2) * np.dot(np.dot(mean_class2,np.linalg.inv(S)),np.transpose(mean_class2)) + np.log(prior[:,0])

print(delta_class1)
print(delta_class2)

if delta_class1 > delta_class2:
    test_class = 1
else:
    test_class = 0
    
print(test_class)

    