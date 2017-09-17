import numpy as np
import pandas as pd
import scipy.signal as sps
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import mean_squared_error
from math import sqrt

head = ["Time","CBFV.C1","ABP","CBFV.C2","End-Tindal","CrCP.C1","RAP.C1","Heart Rate","Sistolic Pressure","Diastolic Pressure","POURCELOT'S INDEX  CHANNEL 1","MEAN SQUARE ERROR FOR P-V CH. 1","CrCP.C2","RAP.C2","SYSTOLIC  VEL. CH.1","DIASTOLIC VEL. CH.1","SYSTOLIC VELOCITY CH. 2","DIASTOLIC VELOCITY CH.2","POURCELOT INDEX CH. 2","MSE FOR P-V  CH. 2"]

#Preprocesamiento: Captura de datos desde csv
#Lectura de los datos de los sujetos en Hipercapnia
def super_preprocesamiento(data_pacientes,remuestreo):
    abp = np.array(pacient01H["ABP"])
    cbfv = np.array(pacient01H["CBFV.C1"])
    abp1 = np.array([])
    cbfv1 = np.array([])
    for i in range(len(abp)):
        if(i%remuestreo == 0 and i != len(abp)-1):
            newabp1 = np.sum(abp[i:i+remuestreo])/remuestreo
            #print("Promedio de abp ", abp[i:i+remuestreo])
            newcbfv1 = np.sum(cbfv[i:i+remuestreo])/remuestreo
            abp1 = np.append(abp1,newabp1)
            cbfv1 = np.append(cbfv1,newcbfv1)
    return abp1,cbfv1

def super_retrasocosmico(data):
    array1 = []
    array2 = []
    array3 = []
    for i in range(0,len(data)-3):
        array1.append(data[i])
    for i in range(1,len(data)-2):
        array2.append(data[i])
    for i in range(2,len(data)-1):
        array3.append(data[i])
    array = [array1,array2,array3]
    array = np.array(array)
    return array.T

def calculate_loss1(X,yprima,y,model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    #print("Yprima")
    #print(yprima)
    #print("Y")
    #print(y)
    #print("W1")
    #print(W1)
    #print("b1")
    #print(b1)
    #print("W2")
    #print(W2)
    #print("b2")
    #print(b2)
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = expit(z1)
    z2 = a1.dot(W2) + b2
    yprima = expit(z2)
    #print("z1")
    #print(z1)
    #print("a1")
    #print(a1)
    #print("z2")
    #print(z2.T)
    # Calculating the loss
    error = (y-yprima)[0].T
    for i in range(len(error)):
        error[i] = (error[i])**2
    data_loss = np.mean(error)
    # Add regulatization term to loss (optional)
    return data_loss

def calculate_loss2(X,output,y,model):
    data_loss = abs(y-output)[0].T
    return data_loss
#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    derivada = np.ndarray((len(x),1))
    for i in range(derivada.shape[0]):
        for j in range(derivada.shape[1]):
            derivada[i,j] = x[i,j]*(1-x[i,j])
    return derivada

def build_model(X,y,inputlayer_neurons,hiddenlayer_neurons,output_neurons,lr,num_passes=10,update_loss=2,print_loss = True):
    #weight and bias initialization
    wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
    bh = np.random.uniform(size=(1,hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
    bout = np.random.uniform(size=(1,output_neurons))
    model = {}
    for i in range(num_passes):
        #Forward Propogation
        hidden_layer_input     = np.dot(X,wh) + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1     = np.dot(hiddenlayer_activations,wout)
        output_layer_input      = output_layer_input1+ bout
        output                  = sigmoid(output_layer_input)
        #print("hidden_layer_input1",hidden_layer_input1.shape)
        #print("hidden_layer_input",hidden_layer_input.shape)
        #print("hiddenlayer_activations",hiddenlayer_activations.shape)
        #print("output_layer_input1",output_layer_input1.shape)
        #print("output_layer_input",output_layer_input.shape)
        #print("output",output.shape)

        #Backpropagation
        E = (y-output)[0].T
        slope_output_layer = derivatives_sigmoid(output)
        #print("slope_output_layer",slope_output_layer.shape)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        #print("slope_hidden_layer",slope_hidden_layer.shape)
        d_output = np.ndarray((len(E),1))
        for j in range(len(E)):
            d_output[j,0] = E[j,0]*slope_output_layer[j,0]
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        model = { 'W1': wh, 'b1': bh, 'W2': wout, 'b2': bout}
        if i%update_loss == 0:
            print("Iteracion %i ERMS %f"%(i,calculate_loss1(X,output,y,model)))
            print("Iteracion %i Error %f"%(i,calculate_loss2(X,output,y,model)))

            #print("y")
            #print(y)
            #print("output")
            #print(output)
        wout += hiddenlayer_activations.T.dot(d_output) *lr
        bout += np.sum(d_output, axis=0,keepdims=True) *lr
        wh += X.T.dot(d_hiddenlayer) *lr
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    return model

#Lectura de datos
pacient01H = pd.read_csv("Data\Hipercapnia\HC090161.csv", sep =";", header = None,names = head)
pacient01N = pd.read_csv("Data\Normocapnia\HC090101.csv", sep =";", header = None,names = head)

muestras = 3
#Se remuestrean los datos
abp1H,cbfv1H = super_preprocesamiento(pacient01H,muestras)
abp1N,cbfv1N = super_preprocesamiento(pacient01N,muestras)

#abp1H = (abp1H-np.min(abp1H))/(np.max(abp1H)-np.min(abp1H))
#abp1N = (abp1N-np.min(abp1N))/(np.max(abp1N)-np.min(abp1N))
#cbfv1H = (cbfv1H-np.min(cbfv1H))/(np.max(cbfv1H)-np.min(cbfv1H))
#cbfv1N = (cbfv1N-np.min(cbfv1N))/(np.max(cbfv1N)-np.min(cbfv1N))

#Se le realizan tres retrasos a las senales de entrada
abp1H = np.matrix(super_retrasocosmico(abp1H))
abp1N = np.matrix(super_retrasocosmico(abp1N))

print("## Hipercapnia ##")
modelH = build_model(abp1H,cbfv1H[range(len(cbfv1H)-3)],3, 150, 1, lr = 0.2, update_loss=2,num_passes=100, print_loss = True)
print("## Normocapnia ##")
modelN = build_model(abp1N,cbfv1N[range(len(cbfv1N)-3)],3, 150, 1, lr = 0.2, update_loss=2,num_passes=100, print_loss = True)

#build_model(abp1N,cbfv1N,50, 3, 1, len(abp1N), learning_rate = 0.12, update_loss=2,num_passes=10, print_loss = True)
