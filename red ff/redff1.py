import numpy as np
import pandas as pd
import scipy.signal as sps
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from scipy.special import expit
from math import sqrt

head = ["Time","CBFV.C1","ABP","CBFV.C2","End-Tindal","CrCP.C1","RAP.C1","Heart Rate","Sistolic Pressure","Diastolic Pressure","POURCELOT'S INDEX  CHANNEL 1","MEAN SQUARE ERROR FOR P-V CH. 1","CrCP.C2","RAP.C2","SYSTOLIC  VEL. CH.1","DIASTOLIC VEL. CH.1","SYSTOLIC VELOCITY CH. 2","DIASTOLIC VELOCITY CH.2","POURCELOT INDEX CH. 2","MSE FOR P-V  CH. 2"]

#Preprocesamiento: Captura de datos desde csv
#Lectura de los datos de los sujetos en Hipercapnia
pacient01H = pd.read_csv("Data\Hipercapnia\HC090161.csv", sep =";", header = None,names = head)
pacient01HABP = sps.resample(pacient01H["ABP"], len(pacient01H["ABP"]/2))
pacient01HABP  = np.matrix(pacient01HABP)
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
    errors = []
    y = np.matrix(y).T
    num_examples = len(X)
    for i in range(0,num_passes):
        print("****************Calculo del forward propagation*********\n")
        print("X.shape: ",X.shape)
        print("W1.shape: ",W1.shape)
        #hidden_layer_input= matrix_dot_product(X,wh) + bh
        z1 = X.dot(W1) + b1
        print("Z1.shape: ",z1.shape)

        #hiddenlayer_activations = activation(hidden_layer_input)
        a1 = np.array(expit(z1))
        print("a1.shape: ",a1.shape)

        #output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout
        z2 = a1.dot(W2) + b2
        print("Z2.shape: ",z2.shape)

        #output = sigmoid(output_layer_input)
        yestimada = np.array(expit(z2))
        yderivada = yestimada*(1-yestimada)
        #E = y-output
        print("Ypyestimadarima.shape: ",yestimada.shape)
        print("Y.shape: ",y.shape)

        error1 = mean_sq
        error = yestimada-y
        print("error.shape: ", error.shape)

        print("\n*******************Calculo de Backpropagation*****************\n")
        delta3 = np.empty_like(error)
        for i in range(len(error)):
            delta3[i] = error[i]*yderivada[i]
        print("delta3.shape",delta3.shape)

        dW2 = (a1.T).dot(delta3)
        print("dw2.shape",dW2.shape)

        #LISTO
        db2 = np.sum(delta3)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)).T
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        #wout = wout + matrix_dot_product(hiddenlayer_activations.Transpose, d_output)*learning_rate
        W1 += -alfa * dW1
        W2 += -alfa * dW2

        #bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate
        #b1 += -alfa * db1
        #b2 += -alfa * db2

        #model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model,errors,num_passes

model,error,num_passes = generate_mlp(pacient01HABP.T,pacient01HCBFV.T,10,1,2,1,0.1)

x = np.linspace(1,num_passes)

plt.plot(error)
plt.show()
