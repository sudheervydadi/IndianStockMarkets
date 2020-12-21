# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:23:32 2020

@author: sudhe
"""

import numpy as np

class Network:
    def __init__(self):
        
        self.network_chain=None
        self.input_feature_count=None
        self.output_feature_count=None
        self.activation_chain=None
        self.activations=list()
        
        self.linear_weights=list()
        self.linear_bias=list()
        self.gradient_weights=list()
        self.gradient_bias=list()
        
        self.non_linear_weights=list()
        self.non_linear_bias=list()
        
        self.Xtrain_data=None
        self.Ytrain_data=None
        self.Xvalidation_data=None
        self.Yvalidation_data=None
        
        
    def input_layer(self,input_data,activation='relu'):
        import numpy as np
        self.input_feature_count=input_data.shape[1]
        self.network_chain=list([np.empty(self.input_feature_count,)])
        self.Xtrain_data=input_data
        self.activation_chain=list([activation])
        
        
    def add_layer(self,number_of_neurons,activation='relu'):
        self.network_chain.append(np.empty(number_of_neurons,))
        self.activation_chain.append(activation)
        
    def hidden_neurons_list(self,hidden_network_list,activation_list):
        for i in hidden_network_list:
            self.network_chain.append(np.empty(i,))
            np.append(self.network_chain.append,activation_list)
    
    def output_layer(self,output_data,activation='sigmoid'):
        self.output_feature_count=output_data.shape[1]
        self.network_chain.append(np.empty(self.output_feature_count,))
        self.Ytrain_data=output_data
        self.activation_chain.append(activation)
        
    
    def intialize_weights_and_bias(self):
        import numpy as np
        for i in range(0,len(self.network_chain)-1):
            self.linear_weights.append(np.random.rand(self.network_chain[i].shape[0],self.network_chain[i+1].shape[0]))
            if(i==len(self.network_chain)-2):
                self.linear_weights.append(np.random.rand(self.network_chain[i+1].shape[0],1))
        
        for i in range(0,len(self.network_chain)-1):
            self.linear_bias.append(np.random.rand(self.network_chain[i+1].shape[0],1))
    
    
    def validation_split(self,validation_size,random_state):
        from sklearn.model_selection import train_test_split
        self.Xtrain_data,self.Xvalidation_data,self.Ytrain_data,self.Yvalidation_data=train_test_split(self.Xtrain_data,self.Ytrain_data,test_size=validation_size, random_state=42)
        
    
    def multiplication(x,w):
        return x.dot(w)
    def addition(weight,bias):
        return weight+bias
    
    def relu(x):
        return np.where(x<0,0,x)
    
    def sigmoid(x):
        np.apply_along_axis(lambda y:1 / (1 + np.exp(-y)),1,x)
        
    
    
    def predict(self,data):
        prediction=data
        for i in range(len(self.linear_weights)-1):
            prediction=np.apply_along_axis(multiplication,1,temp,self.linear_weights[i])
            prediction=np.apply_along_axis(addition,1,temp,self.linear_bias[i])
        
        return prediction
    
        
    def fit(self,epochs=100,batchsize=1,learning_rate=0.001):
        pass
        

output_data=np.ones(shape=(100,1))*2*5

input_data=np.ones(shape=(100,5))*2

model=Network()

model.input_layer(input_data)
model.add_layer(10)
model.add_layer(15)
model.add_layer(8)
model.output_layer(output_data)

model.intialize_weights_and_bias()
model.validation_split(validation_size=0.15, random_state=42)


for i in model.network_chain:
    print(i.shape)



def multiplication(x,w):
    return x.dot(w)
def addition(weight,bias):
    return weight+bias



temp=model.Xtrain_data

for i in range(len(model.linear_weights)-1):
    print(model.linear_bias[i].shape)
    print(np.apply_along_axis(multiplication,1,temp,model.linear_weights[i]).shape)
    temp=np.apply_along_axis(multiplication,1,temp,model.linear_weights[i])
 




for i in range(len(model.linear_weights)-1,-1,-1):
    print(model.linear_weights[i].shape)
    
for i in range(len(model.linear_bias)-1,-1,-1):
    print(model.linear_bias[i].shape)

def relu(i):
    return np.where(i>0,i,0)

def sigmoid(i):
    return np.apply_along_axis(lambda y:1 / (1 + np.exp(-y)),0,i)
    
for i in model.Xvalidation_data:
    print(np.apply_along_axis(relu,0,i).shape)
    break
    for j in range(0,len(model.linear_weights)):
        
        if j>=1:
            print(i.reshape(i.shape[0],1)*model.linear_weights[j]+model.linear_bias[j-1])
            break

model.activation_chain

for i in model.linear_weights:
    print(i.shape)
    
for i in model.linear_bias:
    print(i.shape)
        
        
import tensorflow as tf
a = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
b = tf.keras.activations.sigmoid(a)
b.numpy()

print(sigmoid([-20, -1.0, 0.0, 1.0, 20]))


for k in model.Xtrain_data:
    temp=k
    for i,j in zip(range(0,len(model.activation_chain)),model.activation_chain):
        if i==0:
            if model.activation_chain[i]=='relu':
                model.activatoins.append(relu(temp))
        




