import numpy as np
import pandas as pd
import scipy.signal as sps
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from scipy.special import expit

head = ["Time","CBFV.C1","ABP","CBFV.C2","End-Tindal","CrCP.C1","RAP.C1","Heart Rate","Sistolic Pressure","Diastolic Pressure","POURCELOT'S INDEX  CHANNEL 1","MEAN SQUARE ERROR FOR P-V CH. 1","CrCP.C2","RAP.C2","SYSTOLIC  VEL. CH.1","DIASTOLIC VEL. CH.1","SYSTOLIC VELOCITY CH. 2","DIASTOLIC VELOCITY CH.2","POURCELOT INDEX CH. 2","MSE FOR P-V  CH. 2"]

#Preprocesamiento: Captura de datos desde csv
#Lectura de los datos de los sujetos en Hipercapnia
pacient01H = pd.read_csv("Data\Hipercapnia\HC090161.csv", sep =";", header = None,names = head)
pacient01HABP  = np.matrix(pacient01H["ABP"])
pacient01HCBFV = pacient01H["CBFV.C1"]

def generate_mlp(X,y,num_passes, Ne,Nc,Ns,alfa):
    #We initialize weights and biases with random values
    #(This is one time initiation. In the next iteration, we will use updated weights, and biases). Let us define:
    print("Generacion aleatoria de los pesos de la capa de entrada y capa oculta")
    np.random.seed(0)
    W1 = np.random.randn(Ne, Nc) / np.sqrt(Ne)
    b1 = np.zeros((1, Nc))
    W2 = np.random.randn(Nc, Ns) / np.sqrt(Nc)
    b2 = np.zeros((1, Ns))
    model = {}
    y = np.matrix(y).T
    for i in range(0,num_passes):
        print("****************Calculo del forward propagation*********\n")
        print("X.shape: ",X.shape)
        print("W1.shape: ",W1.shape)
        #hidden_layer_input= matrix_dot_product(X,wh) + bh
        z1 = X.dot(W1) + b1
        print("Z1.shape: ",z1.shape)

        #hiddenlayer_activations = sigmoid(hidden_layer_input)
        a1 = expit(z1)
        print("a1.shape: ",a1.shape)

        #output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout
        z2 = a1.dot(W2) + b2
        print("Z2.shape: ",z2.shape)

        #output = sigmoid(output_layer_input)
        yprima = np.array(expit(z2))

        #E = y-output
        print("Yprima.shape: ",yprima.shape)
        print("Y.shape: ",y.shape)

        error = yprima-y
        print("ERROR GENERAL ", error)
        print("Error.shape", error.shape)
        print("*******************Calculo del Backpropagation*****************\n")
        #slope_output_layer = derivatives_sigmoid(output)
        slope_output_layer = yprima*(1-yprima)
        print("slope_output_layer.shape", slope_output_layer.shape)

        #slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        slope_hidden_layer = a1*(a1-1)

        #d_output = E * slope_output_layer
        d_output = error * slope_output_layer

        #Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)
        error_at_hidden_layer = d_output.dot(W2)

        #d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        d_hiddenlayer = error_at_hidden_layer*slope_hidden_layer

        #wout = wout + matrix_dot_product(hiddenlayer_activations.Transpose, d_output)*learning_rate
        W2 = W2 + yprima.T.dot(d_output)*alfa
        W1 = W1 +X.T.dot(d_hiddenlayer)*alfa

        #bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate
        b1 = b1+np.sum(d_hiddenlayer,axis = 0)*alfa
        b2 = b2+np.sum(d_output,axis = 0)*alfa

        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

generate_mlp(pacient01HABP.T,pacient01HCBFV.T,10,1,2,1,0.1)
