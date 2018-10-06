# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:24:06 2018

@author: 18766139
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from numpy import empty




with open('2018_CI_Assignment_Training_Data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data = [r for r in readCSV]
    data.pop(0)
    
array=np.asanyarray(data,dtype='f')
#p=array[:,6]

plt.plot(array[:,6],'o')
plt.title('Training data with outliner')
plt.show()
#remove outliner
#%%
Q1=np.percentile(array[:,6], 25);  # the value 25 is fixed for every problem;  
Q3=np.percentile(array[:,6], 75); # the value 75 is fixed for every problem; 
range=[Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)];
position=np.concatenate((np.where(array[:,6]>range[1]),np.where(array[:,6]<range[0])),axis=1) # np.concatenate is used for combining arrays
p_new= np.delete(array, position,axis=0)   # np.delete is used for removing some elements in certain positions/places in a list
oo=(p_new.shape)
len(p_new)
plt.plot(p_new[:,6],'o')
plt.title('Training data without outliner')
plt.ylim(0, 500)
plt.show()
outliners=[]
p_new_max=np.max(p_new[:,6]) #get max value of new data set 

count=0
count1=0
while count<p_new.shape[0]:
    
    if(array[count,6]>p_new_max): #create list of ouliners
        outliners.append(array[count,6])
        count1+=1
    count+=1
print(outliners)
#with open("New_training_data.csv", "w") as newcsvfile:
 #   writer = csv.writer(newcsvfile)
  #  writer = csv.writer(newcsvfile)
#writer.writerow(p_new)
#newcsvfile.close()

with open('2018_CI_Assignment_Testing_Data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    dataTesting = [r for r in readCSV]
    dataTesting.pop(0)
    
arrayTesting=np.asanyarray(dataTesting,dtype='f')
#p=array[:,6]

plt.plot(arrayTesting[:,6],'o')
plt.title('Training data with outliner')
plt.show()

Q1=np.percentile(arrayTesting[:,6], 25);  # the value 25 is fixed for every problem;  
Q3=np.percentile(arrayTesting[:,6], 75); # the value 75 is fixed for every problem; 
range=[Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)];
position=np.concatenate((np.where(arrayTesting[:,6]>range[1]),np.where(arrayTesting[:,6]<range[0])),axis=1) # np.concatenate is used for combining arrays
p_new_testing= np.delete(arrayTesting, position,axis=0)   # np.delete is used for removing some elements in certain positions/places in a list
#oo=(p_new_testing.shape)
len(p_new_testing)
plt.plot(p_new_testing[:,6],'o')
plt.title('Training data without outliner')
plt.ylim(0, 500)
plt.show()
outlinersTesting=[]
p_new_max=np.max(p_new_testing[:,6]) #get max value of new data set 

count=0

while count<p_new_testing.shape[0]:
    
    if(arrayTesting[count,6]>p_new_max): #create list of ouliners
        outlinersTesting.append(arrayTesting[count,6])
        
    count+=1
print(outlinersTesting)


#create a input maxtrix for correlation analysis
inputMatrix=np.column_stack(p_new)
#x1=np.linspace(1, 10, 100)

#calculate the Correlation Coefficient Matrix
CCM=np.corrcoef(inputMatrix)
print('Correlation analysis:')
print(CCM)
print()
print(CCM.shape)

#based on coef analysis the new data set will be (T(t-2),D(t),P(t+1))
new_data_set=p_new[:, [0,5, 6]]
#data_bin=20;
plt.figure()

plt.hist(new_data_set[:,0], 3) # histogram for input1 
plt.title('Histogram of input 1')
plt.ylim(0, 500)
plt.show() 
plt.hist(new_data_set[:,1], 5) # histogram for input2 
plt.title('Histogram of input 2')
plt.show() 
plt.hist(new_data_set[:,2], 3) # histogram for output 
plt.title('Histogram of output')
plt.show() 





temperature= ctrl.Antecedent(np.arange(16,37,1),'temperature')
demand=ctrl.Antecedent(np.arange(3700,6900,1),'demand')
price=ctrl.Consequent(np.arange(10,55,1),'price')
#temperature= ctrl.Antecedent(new_data_set[:,0],'temperature')
#demand=ctrl.Antecedent(new_data_set[:,1],'demand')
#price=ctrl.Consequent(new_data_set[:,2],'price')


#temperature.automf(3);
#demand.automf(5);
#price.automf(3);
a1=temperature['low']=fuzz.trimf(temperature.universe,[16,16,27])
a2=temperature['warm']=fuzz.trimf(temperature.universe,[16,27,37])
a3=temperature['hot']=fuzz.trimf(temperature.universe,[27,37,37])

degreeOfInput1=np.column_stack((a1,a2,a3))

temperature.view()

b1=demand['very little']=fuzz.trimf(demand.universe,[3700,3700,4250])
b2=demand['little']=fuzz.trimf(demand.universe,[3700,4250,5250])
b3=demand['average']=fuzz.trimf(demand.universe,[4250,5250,6500])
b4=demand['high']=fuzz.trimf(demand.universe,[5250,6500,6900])
b5=demand['very high']=fuzz.trimf(demand.universe,[6500,6900,6900])

degreeOfInput2=np.column_stack((b1,b2,b3,b4,b5))
demand.view()

c1=price['low']=fuzz.trimf(price.universe,[0,10,23])
c2=price['medium']=fuzz.trimf(price.universe,[10,23,35])
c3=price['high']=fuzz.trimf(price.universe,[23,35,55])

degreeOfOput=np.column_stack((c1,c2,c3))
price.view()
rule1= ctrl.Rule(temperature['low']& demand['very little'],price['low'])
rule2= ctrl.Rule(temperature['low']& demand['little'],price['medium'])
rule3= ctrl.Rule(temperature['low']& demand['average'],price['medium'])
rule4= ctrl.Rule(temperature['low']& demand['high'],price['medium'])
rule15= ctrl.Rule(temperature['low']& demand['very high'],price['high'])
rule5= ctrl.Rule(temperature['warm']& demand['very little'],price['low'])
rule6= ctrl.Rule(temperature['warm']& demand['little'],price['medium'])
rule7= ctrl.Rule(temperature['warm']& demand['average'],price['medium'])
rule8= ctrl.Rule(temperature['warm']& demand['high'],price['medium'])
rule9= ctrl.Rule(temperature['warm']& demand['very high'],price['high'])
rule10= ctrl.Rule(temperature['hot']& demand['very little'],price['low'])
rule11= ctrl.Rule(temperature['hot']& demand['little'],price['medium'])
rule12= ctrl.Rule(temperature['hot']& demand['average'],price['medium'])
rule13= ctrl.Rule(temperature['hot']& demand['high'],price['high'])
rule14= ctrl.Rule(temperature['hot']& demand['very high'],price['high'])
rule1.view()

pricing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12,rule13, rule14,rule15])
pricing = ctrl.ControlSystemSimulation(pricing_ctrl)



#%%
pricing.input['temperature']=23.4
pricing.input['demand']=4413.1
pricing.compute()
print (pricing.output['price'])
price.view(sim=pricing)



#%%
#System out put
count=0
intput1=new_data_set[:,0]
intput2=new_data_set[:,1]
targetOutput=new_data_set[:,2]

System_outputs=np.zeros(901,dtype=np.float64)
while count<901:
    pricing.input['temperature']= intput1[count]
    pricing.input['demand']=intput2[count]
    pricing.compute()
    System_outputs[count]=pricing.output['price']    ### Save each output in the object 'System_outputs'
    count+=1

#%%
#print(System_outputs)  # should have three values/elements
System_outputs
Err=np.sum(np.absolute(targetOutput-System_outputs)/np.absolute(targetOutput))/901 
print('The Average Relative Error Value is', Err)



plt.cla
plt.clf
plt.figure()
plt.plot(targetOutput)
plt.plot(System_outputs)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('')
plt.legend(['Target Outputs', 'System Outputs'])
plt.show()

#testing dataset
count=0
intput1=p_new_testing[:,0]
intput2=p_new_testing[:,5]
targetOutput=p_new_testing[:,6]

System_outputs=np.zeros(485,dtype=np.float64)
while count<485:
    pricing.input['temperature']= intput1[count]
    pricing.input['demand']=intput2[count]
    pricing.compute()
    System_outputs[count]=pricing.output['price']    ### Save each output in the object 'System_outputs'
    count+=1
System_outputs
Err=np.sum(np.absolute(targetOutput-System_outputs)/np.absolute(targetOutput))/485 
print('The Average Relative Error Value is', Err)
plt.cla
plt.clf
plt.figure()
plt.plot(targetOutput)
plt.plot(System_outputs)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('')
plt.legend(['Target Outputs', 'System Outputs'])
plt.show()
x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])
defuzz_centroid = fuzz.defuzz(new_data_set[:,2], price, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x, mfx, 'bisector')
defuzz_mom = fuzz.defuzz(x, mfx, 'mom')
defuzz_som = fuzz.defuzz(x, mfx, 'som')
defuzz_lom = fuzz.defuzz(x, mfx, 'lom')
