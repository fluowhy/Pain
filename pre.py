# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.stats import kurtosis
import scipy.signal as sc

# Este programa calcula la potencia en distintas bandas para EEG de tres canales antes del estimulo
# FALTA:	guardar estimulo dado y percepcion del individuo, de esa manera, se puede calcular el promedio por estimulo dado y diferenciarlo del estimulo percibido. 
# 		la idea es obtener una tabla de la siguiente estructura:
# |individuo|potencia banda x|...|nivel de analgesia|estimulo||percepcion|| 
# la ultima columna es la clase a predecir
# se pueden agregar caracteristicas de la señal como curtosis, varianza, cross frequency coupling


def rangos(A):
	n = A.shape[0]
	m = A.shape[1]
	for i in range(n):
		for j in range(m):
			var = A[i, j, 1]
			if var==0 or var==1:
				A[i, j, 1] = 1
			if var==2 or var==3:
				A[i, j, 1] = 2
			if var==4 or var==5:
				A[i, j, 1] = 3
			if var>5:
				A[i, j, 1] = 4
	return A
#data = np.loadtxt('/home/mauricio/Documents/Uni/Neuro/data/SA001/SA001_A1_7.txt')

directory = '/home/mauricio/Documents/Pain/'

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
EPD = [] # estimulo-percepcion dolor 
EPA = [] # estimulo-percepcion analgesia
# recorre datos de dolor y crea tensor:
# SD: persona, canal, estimulo, muestra
# SA: persona, canal, nivel analgesia, estimulo, muestra

print 'recorro personas'
for k in folders:
	
	sd = []
	sa = []
	for j in [7, 8, 9]:
		data = []
		dataA = []
		for i in [1, 2]:
			data.append(np.loadtxt(directory+'Data/SA00'+str(k)+'/SA00'+str(k)+'_D'+str(i)+'_'+str(j)+'.txt'))
		data = np.concatenate((data[0], data[1]))			
		sd.append(data)		
		for i in [1, 2, 3]:
			data = np.loadtxt(directory+'Data/SA00'+str(k)+'/SA00'+str(k)+'_A'+str(i)+'_'+str(j)+'.txt')
			dataA.append(data)
		dataA = np.array(dataA)
		sa.append(dataA)
	sd = np.array(sd)	
	SD.append(sd)	
	sa = np.array(sa)
	SA.append(sa)
SD = np.array(SD)
SA = np.array(SA)

# se recorren nuevamente para guardar estimulo-percepcion
# sujeto, estimulo, percepcion
print 'nuevamente'
for k in folders:
	
	epd = []
	epa = []
	for i in [1, 2]:
		dt = np.loadtxt(directory+'Data/SA00'+str(k)+'/SA00'+str(k)+'_D'+str(i)+'.txt')
		epd.append(dt)
	EPD.append(np.concatenate((epd[0], epd[1])))
	for i in [1, 2, 3]:
		dt = np.loadtxt(directory+'Data/SA00'+str(k)+'/SA00'+str(k)+'_A'+str(i)+'.txt')
		epa.append(dt)
	EPA.append(np.concatenate((epa[0], epa[1], epa[2])))
EPA = np.array(EPA)
EPD = np.array(EPD)

# transformacion sujeto 6 (arreglo):

for i in range(len(EPD[3,:,0])):
	j = EPD[3, i, 0]
	if j==128:
		EPD[3, i, 0] = 0
	elif j==139:
		EPD[3, i, 0] = 11
	elif j==149:
		EPD[3, i, 0] = 21
	elif j==159:
		EPD[3, i, 0] = 31
	elif j==169:
		EPD[3, i, 0] = 41

# transforma percepcion 2 digitos a 1 digito	

EPD[:, :, 0] = (EPD[:, :, 0] - 1)/10
EPA[:, :, 0] = (EPA[:, :, 0] - 1)/10

# rangos de percepcion

EPD = rangos(EPD)
EPA = rangos(EPA)

# saca el promedio de la serie

print 'resto el promedio'
SD = (SD.T-np.mean(SD, axis=3).T).T
SA = (SA.T-np.mean(SA, axis=4).T).T

# promedio canales
# SD: persona, estimulo, muestra
# SA: persona, nivel analgesia, estimulo, muestra

SD = np.sum(SD, axis=1)/3
SA = np.sum(SA, axis=1)/3

# promedia la señal de cada sujeto por percepcion

S = [] # sujeto, nivel de analgesia, percepcion (creciente 1-4), muestra
for i in range(len(folders)):
	s = []
	lvl = []
	for j in range(4): # dolor, recorre niveles de percepcion
		argd = np.nonzero(EPD[i, :, 1]==j+1)[0]		
		if len(argd)!=0:			
			prom_d = np.sum(SD[i, :, :][argd, :], axis=0)/len(argd)
		elif len(argd)==0:
			prom_d = np.zeros(2048)
		lvl.append(prom_d)
	lvl = np.array(lvl)	
	s.append(lvl)	
	for k in range(3): # niveles analgesia
		lvl = []	
		for j in range(4):
			arga = np.nonzero(EPA[i, k*20:(k + 1)*20, 1])[0]
			prom_a = np.sum(SA[i, k, :, :][arga, :], axis=0)/len(arga)
			lvl.append(prom_a)
		lvl = np.array(lvl)
		s.append(lvl)
	s = np.array(s)
	S.append(s)		
S = np.array(S)

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
for i in range(len(F)):
	w, h = sc.freqz(B[i], A[i], worN=1000)
	plt.plot((fs * 0.5 / np.pi) * w, 20*np.log10(abs(h)), label=bandas[i])
plt.xlim([0, 150])
plt.ylim([-30, 5])
plt.xlabel('frecuencia Hz')
plt.ylabel('potencia dB')
plt.legend()
plt.show()
"""

Y = []
for a, b in zip(A, B):
	Y.append(sc.filtfilt(b, a, S, axis=3))
Y = np.array(Y) # shape: banda, individuo, nivel de analgesia percepcion, muestra
"""
# plots de señales filtradas
plt.plot(S[0,0,0,:], label='raw', linewidth=0.25)
for i in range(5):
	plt.plot(Y[i,0,0,0,:], label=bandas[i])
plt.xlabel('muestra')
plt.ylabel('amplitud')
plt.legend()
plt.show()
"""
"""
# FFT
print 'calculo transformada de Fourier'
tfS = np.abs(np.fft.fft(S[0,0,0,:]))
tfy = np.abs(np.fft.fft(Y[1,0,0,0,:]))
freq = np.fft.fftfreq(2048, 1./fs)
plt.plot(freq, 20*np.log10(tfS))
plt.plot(freq, 20*np.log10(tfy))
plt.show()
"""

# calculo de la potencia en cada banda en dB
print 'calculo la potencia de cada banda en dB'
P = 10*np.log10(np.trapz(Y**2, t, axis=4))/1. # T=1 segundo

# construye base de datos
# individuo, potencia en banda, nivel analgesia, percepcion
# 0,         1, 2, 3, 4, 5,     6,               7
print 'construyendo base de datos'
BD = []
for j in range(7): # numero de individuos
	BD.append(np.array([j, p[0]]))
BD = np.array(BD)
np.savetxt('database', BD)

