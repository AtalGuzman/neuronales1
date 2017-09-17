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


def build_model(X,y,nn_hdim, nn_input_dim, nn_output_dim, num_examples, learning_rate, update_loss, print_loss=False, num_passes=1):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(71)

    W1 = 0.1*np.random.randn(nn_input_dim, nn_hdim)/np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = 0.1*np.random.randn(nn_hdim, nn_output_dim)/np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    #print("W1 shape", W1.shape)
    #print("b1 shape", b1.shape)
    #print("W2 shape", W2.shape)
    #print("b2 shape", b2.shape)

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = expit(z1)
        z2 = a1.dot(W2) + b2
        #SALIDA
        yprima = z2
        #print("FORWARD STAGE")
        #print("z1 shape",z1.shape)
        #print("z1 shape",z1.shape)
        # Backpropagation
        #slope_output_layer = derivatives_sigmoid(output)
        error = np.matrix(y[range(len(y)-3)]-yprima)[0].T #E = y-output

        slope_output_layer = yprima
        slope_hidden_layer = np.ndarray((len(X), nn_hdim))
        d_output = []
        for j in range(len(slope_output_layer)): #DERIVADA DE LA SALIDA
            slope_output_layer[j] = 1 #slope_output_layer = derivatives_sigmoid(output)
            d_output.append(error[j,0]*slope_output_layer[j,0])  #d_output = E * slope_output_layer
            for k in range(len(a1[j])):
                #print("valor pendiente %f"%(1-a1[j,k]**2))
                slope_hidden_layer[j,k] = a1[j,k]*(1-a1[j,k])      #slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)                                  #

        slope_hidden_layer = np.array(slope_hidden_layer)
        d_output = np.matrix(d_output).T

        #print("slope_hidden_layer.shape ", slope_hidden_layer.shape)
        #print("slope_output_layer.shape ", slope_output_layer.shape)
        #print("d_output.shape ", d_output.shape)

        #Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)
        Error_at_hidden_layer = (d_output).dot(W2.T)
        #print("Error_at_hidden_layer.shape ", Error_at_hidden_layer.shape)

        d_hiddenlayer = np.ndarray((len(X), nn_hdim))
        for j in range(len(Error_at_hidden_layer)):
            for k in range(len(Error_at_hidden_layer[j])):
                d_hiddenlayer[j,k] = (Error_at_hidden_layer[j,k]*slope_hidden_layer[j,k])
        #print("d_hiddenlayer.shape ", d_hiddenlayer.shape)

        dW2 = (a1.T).dot(d_output)*learning_rate  #dwout = matrix_dot_product(hiddenlayer_activations.Transpose, d_output)*learning_rate
        #print("dW2.shape ", dW2.shape)

        dW1 = np.dot(X.T,d_hiddenlayer)*learning_rate #dwhiden = dmatrix_dot_product(X.Transpose,d_hiddenlayer)*learning_rate
        #print("dW1.shape ", dW1.shape)

        db1 = np.sum(d_hiddenlayer, axis=0, keepdims=True)*learning_rate
        #print("db1.shape ", db1.shape)
        #d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        db2 = np.sum(d_output, axis=0)*learning_rate
        #print("db2.shape ", db2.shape)

        # Gradient descent parameter update
        #model(s)
        W1 = W1+dW1
        b1 = b1+db1
        W2 = W2+dW2
        b2 = b2+db2
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if print_loss and i%update_loss == 0:
            print("#####    Loss after iteration %i: %f   ####"%(i, calculate_loss1(X,yprima,y[range(len(y)-3)],model)))
        #print("ERROR")
        #print(error)
    return model

#Lectura de datos
pacient01H = pd.read_csv("Data\Hipercapnia\HC090161.csv", sep =";", header = None,names = head)
pacient01N = pd.read_csv("Data\Normocapnia\HC090101.csv", sep =";", header = None,names = head)

muestras = 3
#Se remuestrean los datos
abp1H,cbfv1H = super_preprocesamiento(pacient01H,muestras)
abp1N,cbfv1N = super_preprocesamiento(pacient01N,muestras)

#Se le realizan tres retrasos a las senales de entrada
abp1H = np.array(super_retrasocosmico(abp1H))

abp1N = np.array(super_retrasocosmico(abp1N))

build_model(abp1H,cbfv1H,50, 3, 1, len(abp1H), learning_rate = 0.12, update_loss=2,num_passes=10, print_loss = True)
#build_model(abp1N,cbfv1N,50, 3, 1, len(abp1N), learning_rate = 0.12, update_loss=2,num_passes=10, print_loss = True)
