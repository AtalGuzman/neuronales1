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
    print(array.shape)
    return array.T

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

    
def build_model(X,y,nn_hdim, nn_input_dim, nn_output_dim, num_examples, epsilon, print_loss=False, num_passes=1):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
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
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        #SALIDA

        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        y = 1/(1+np.exp(-z2))
        #print("FORWARD STAGE")
        #print("z1 shape",z1.shape)
        #print("a1 shape",z1.shape)
        #print("z1 shape",z1.shape)
        # Backpropagation
        delta3 = probs
        print("Delta3.shape: ",delta3.shape)
        delta3[range(num_examples)] -= 1 #DERIVADA DE LA SALIDA
        dW2 = (a1.T).dot(delta3)        #Derivada de la capa oculata
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        #dW2 += reg_lambda * W2
        #dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print("Loss after iteration %i: %f" %(i, calculate_loss(model)))

    return model

#Lectura de datos
pacient01H = pd.read_csv("Data\Hipercapnia\HC090161.csv", sep =";", header = None,names = head)
pacient02H = pd.read_csv("Data\Hipercapnia\HC091161.csv", sep =";", header = None,names = head)
pacient03H = pd.read_csv("Data\Hipercapnia\HC092161.csv", sep =";", header = None,names = head)
pacient04H = pd.read_csv("Data\Hipercapnia\HC093161.csv", sep =";", header = None,names = head)
pacient05H = pd.read_csv("Data\Hipercapnia\HC094161.csv", sep =";", header = None,names = head)

pacient01N = pd.read_csv("Data\Normocapnia\HC090101.csv", sep =";", header = None,names = head)
pacient02N = pd.read_csv("Data\Normocapnia\HC091101.csv", sep =";", header = None,names = head)
pacient03N = pd.read_csv("Data\Normocapnia\HC092101.csv", sep =";", header = None,names = head)
pacient04N = pd.read_csv("Data\Normocapnia\HC093101.csv", sep =";", header = None,names = head)
pacient05N = pd.read_csv("Data\Normocapnia\HC094101.csv", sep =";", header = None,names = head)

muestras = 3
#Se remuestrean los datos
abp1H,cbfv1H = super_preprocesamiento(pacient01H,muestras)
abp2H,cbfv2H = super_preprocesamiento(pacient02H,muestras)
abp3H,cbfv3H = super_preprocesamiento(pacient03H,muestras)
abp4H,cbfv4H = super_preprocesamiento(pacient04H,muestras)
abp5H,cbfv5H = super_preprocesamiento(pacient05H,muestras)

abp1N,cbfv1N = super_preprocesamiento(pacient01N,muestras)
abp2N,cbfv2N = super_preprocesamiento(pacient02N,muestras)
abp3N,cbfv3N = super_preprocesamiento(pacient03N,muestras)
abp4N,cbfv4N = super_preprocesamiento(pacient04N,muestras)
abp5N,cbfv5N = super_preprocesamiento(pacient05N,muestras)

#Se le realizan tres retrasos a las senales de entrada
abp1H = np.array(super_retrasocosmico(abp1H))
abp2H = np.array(super_retrasocosmico(abp2H))
abp3H = np.array(super_retrasocosmico(abp3H))
abp4H = np.array(super_retrasocosmico(abp4H))
abp5H = np.array(super_retrasocosmico(abp5H))

abp1N = np.array(super_retrasocosmico(abp1N))
abp2N = np.array(super_retrasocosmico(abp2N))
abp3N = np.array(super_retrasocosmico(abp3N))
abp4N = np.array(super_retrasocosmico(abp4N))
abp5N = np.array(super_retrasocosmico(abp5N))

build_model(abp1H,cbfv1H,50, 3, 1, len(abp1H), epsilon = 0.2, num_passes=100)
