from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt

#tresh function
unit_step = lambda x: 0 if x < 0 else 1

#data to learn, column 2 is the bias
training_data = [ (array([0,0,1]), 0),
		(array([0,1,1]), 1),
		(array([1,0,1]), 1),
		(array([1,1,1]), 0), ]

w = random.rand(3)
print(w.shape)
errors = []
eta = 0.2		#Factor de aprendizaje
n = 50			#Cantidad de iteraciones

for i in xrange(n):
	#print("inicial w", w)  			#Se muestran el vector de pesos original
	x, expected = choice(training_data) #Se escogen uno de los datos de entrenamiento
	#print("choice data",x)				#Se muestran
	result = dot(w, x)					#Se calcula el resultado del producto punto
	#print("result",result)				#Se muestra el resultado
	error = expected - unit_step(result)#Se calcula el error de la respuesta de la neurona
	errors.append(error)				#Se muestra el error
	#print("error : ",error)			#Se muestra el erro obtenido
	w += eta * error * x				#Se calcula la variacion de pesos
	#print("w after iter",w)			#Se muestran los nuevos

for x, _ in training_data:
	result = dot(x, w)					#Se muestran los resulatdos con cada una de los resultados de la compuerta logica
	print("{}: {} -> {}".format(x[:2], result, unit_step(result))) #Se muestra el resultado

plt1 = plt.plot(xrange(n),errors)
plt.show()
