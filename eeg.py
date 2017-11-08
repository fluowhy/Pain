# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.stats import kurtosis
import scipy.signal as sc

# Este programa calcula la potencia en distintas bandas para EEG de tres canales antes del estimulo
# FALTA:	guardar files desde MATLAB BIEN!
# 		guardar estimulo dado y percepcion del individuo, de esa manera, se puede calcular el promedio por estimulo dado y diferenciarlo del estimulo percibido. 
# 		la idea es obtener una tabla de la siguiente estructura:
# |individuo|potencia banda x|...|nivel de analgesia|estimulo||percepcion|| 
# la ultima columna es la clase a predecir
# se pueden agregar caracteristicas de la señal como curtosis, varianza, cross frequency coupling

data = np.loadtxt('/home/mauricio/Documents/Uni/Neuro/data/SA001/SA001_A1_7.txt')

folders = [1, 3, 4, 6, 7, 8, 9]
fs = 2048
f_delta = np.array([1, 4])
f_theta = np.array([4, 8])
f_alpha = np.array([8, 12])
f_beta = np.array([12, 30])
f_gamma = np.array([30, 100])

F = np.array([f_delta, f_theta, f_alpha, f_beta, f_gamma])
bandas = ['delta', 'theta', 'alpha', 'beta', 'gamma']
t = np.linspace(0, 1, 2048)

SD = [] # dolor
SA = [] # analgesia
# recorre datos de dolor y crea tensor:
# SD: persona, canal, estimulo, muestra
# SA: persona, canal, nivel analgesia, estimulo, muestra

print 'recorro personas'
for k in folders:
	print k
	sd = []
	sa = []
	for j in [7, 8, 9]:
		data = []
		dataA = []
		for i in [1, 2]:
			data.append(np.loadtxt('/home/mauricio/Documents/Uni/Neuro/data/SA00'+str(k)+'/SA00'+str(k)+'_D'+str(i)+'_'+str(j)+'.txt'))
		data = np.concatenate((data[0], data[1]))			
		sd.append(data)		
		for i in [1, 2, 3]:
			data = np.loadtxt('/home/mauricio/Documents/Uni/Neuro/data/SA00'+str(k)+'/SA00'+str(k)+'_A'+str(i)+'_'+str(j)+'.txt')
			dataA.append(data)
		dataA = np.array(dataA)
		sa.append(dataA)
	sd = np.array(sd)	
	SD.append(sd)	
	sa = np.array(sa)
	SA.append(sa)
SD = np.array(SD)
SA = np.array(SA)



# saca el promedio de la serie

print 'resto el promedio'
SD = (SD.T-np.mean(SD, axis=3).T).T
SA = (SA.T-np.mean(SA, axis=4).T).T

# aplica filtro de fase lineal en bandas delta, theta, alpha, beta y gamma
print 'estoy creando los filtros tipo butterworth fase lineal'
B = []
A = []
for f in F:
	b, a = sc.butter(3, f*2/fs, btype='bandpass')
	B.append(b)
	A.append(a)

"""
# visualizacion del corte del filtro

w_d, h_d = sc.freqz(b_d, a_d, worN=1000)
w_t, h_t = sc.freqz(b_t, a_t, worN=1000)
w_a, h_a = sc.freqz(b_a, a_a, worN=1000)
w_b, h_b = sc.freqz(b_b, a_b, worN=1000)
w_g, h_g = sc.freqz(b_g, a_g, worN=1000)
plt.plot((fs * 0.5 / np.pi) * w_d, 20*np.log10(abs(h_d)), label='delta')
plt.plot((fs * 0.5 / np.pi) * w_t, 20*np.log10(abs(h_t)), label='theta')
plt.plot((fs * 0.5 / np.pi) * w_a, 20*np.log10(abs(h_a)), label='alpha')
plt.plot((fs * 0.5 / np.pi) * w_b, 20*np.log10(abs(h_b)), label='beta')
plt.plot((fs * 0.5 / np.pi) * w_g, 20*np.log10(abs(h_g)), label='gamma')
plt.xlim([0, 150])
plt.ylim([-30, 5])
plt.xlabel('frecuencia Hz')
plt.ylabel('potencia dB')
plt.legend()
plt.show()
"""

YD = []
YA = []
for a, b in zip(A, B):
	YD.append(sc.filtfilt(b, a, SD, axis=3))
	YA.append(sc.filtfilt(b, a, SA, axis=4))
YD = np.array(YD) # shape: banda, individuo, canal, estimulo, muestra
YA = np.array(YA) # shape: banda, individuo, canal, nivel de analgesia, estimulo, muestra

# plots de señales filtradas
plt.plot(SD[0,0,0,:], label='raw', linewidth=0.1)
for i in range(5):
	plt.plot(YD[i,0,0,0,:], label=bandas[i])
plt.xlabel('muestra')
plt.ylabel('amplitud')
plt.legend()
plt.show()

"""
# FFT

print 'calculo transformada de Fourier'
tfS = np.abs(np.fft.fft(S, axis=4))
tfy = np.abs(np.fft.fft(y, axis=4))
freq = np.fft.fftfreq(2048, 1/fs)
plt.plot(freq, 20*np.log10(tfS[0,0,0,0,:]))
plt.plot(freq, 20*np.log10(tfy[0,0,0,0,:]))
plt.show()
"""

# calculo de la potencia en cada banda
print 'calculo la potencia de cada banda'
pD = 10*np.log10(np.trapz(YD**2, t, axis=4))
pA = 10*np.log10(np.trapz(YA**2, t, axis=5))

# calculo del promedio de los 3 canales
PD = np.mean(pD, axis=2)
PA = np.mean(pA, axis=2)

# construye base de datos
print 'construyendo base de datos'
BD = []
for j in range(7): # numero de individuos
	for i in range(120):
		
		bd = np.array([j, PD[0,j,i], PD[1,j,i], PD[2,j,i], PD[3,j,i], PD[4,j,i], 0]) # falta estimulo y percepcion
		BD.append(bd)
	for k in range(3): # 3 niveles de analgesia
		for i in range(60):
			bd = np.array([j, PA[0,j,k,i], PA[1,j,k,i], PA[2,j,k,i], PA[3,j,k,i], PA[4,j,k,i], k+1])
			BD.append(bd)
BD = np.array(BD)




