from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #print(xx)
    #print(yy)
    #print(np.c_[xx.ravel(), yy.ravel()])
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    #print(Z)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

#tresh function
unit_step = lambda x: 0 if x < 0 else 1

#data to learn, column 2 is the bias
training_data = [ (array([0,0,1]), 0),
		(array([0,1,1]), 0),
		(array([1,0,1]), 0),
		(array([1,1,1]), 1), ]

w = random.rand(3)
print(w.shape)
errors = []
eta = 0.2		#Factor de aprendizaje
n = 50			#Cantidad de iteraciones

for i in xrange(n):
	print("inicial w", w)  				#Se muestran el vector de pesos original
	x, expected = choice(training_data) #Se escogen uno de los datos de entrenamiento
	print("choice data",x)				#Se muestran
	result = dot(w, x)					#Se calcula el resultado del producto punto
	print("result",result)				#Se muestra el resultado
	error = expected - unit_step(result)#Se calcula el error de la respuesta de la neurona
	errors.append(error)				#Se muestra el error
	print("error : ",error)				#Se muestra el erro obtenido
	w += eta * error * x				#Se calcula la variacion de pesos
	print("w after iter",w)				#Se muestran los nuevos

for x, _ in training_data:
	result = dot(x, w)												#Se muestran los resulatdos con cada una de los resultados de la compuerta logica
	print("{}: {} -> {}".format(x[:2], result, unit_step(result))) 	#Se muestra el resultado

plot_decision_boundary(lambda x: predict(x))

plt1 = plt.plot(xrange(n),errors)
plt.show()
