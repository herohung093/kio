# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:22:11 2018

@author: hungv
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl




with open('2018_CI_Assignment_Training_Data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data = [r for r in readCSV]
    data.pop(0)
    
dataArray=np.asanyarray(data,dtype='f') #save data from training file to array
plt.plot(dataArray[:,6],'o')
plt.title('Training data with outliner')
plt.show()

count=0
while count<(p_new.shape)[0]:
    
    if(p_new[count,6]>p_new_max): #create list of ouliners
        outliners.append(p_new[count,6])
        count+=1
print(outliners)