# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy.stats import kurtosis
import scipy.signal as sc

# Este codigo implementa SVM con datos de eeg para la clasificacion y prediccion de dolor.

BD = np.loadtxt('database')

features = np.array(['number', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'analgesia', 'stimulus', 'perception'])
feat_us = np.array([1, 4, 5, 6, 7])

print 'caracteristicas'
print features[feat_us]
# preprocesamiento

# se retira datos inconclusos/erroneos

#arg0 = np.nonzero(BD[:,7]==0)[0]
#arg1 = np.nonzero(BD[:,7]>100)[0]
#BD = np.delete(BD, [np.concatenate((arg0, arg1))], axis=0)
#BD[:,7] = (BD[:,7]-1)/10

# normalizacion

# rango -1, 1 para potencias
# probar normalizaion gaussiana
"""
minn = np.min(BD[:, [1, 2, 3, 4, 5]], axis=0)
maxx = np.max(BD[:, [1, 2, 3, 4, 5]], axis=0)
BD[:, [1, 2, 3, 4, 5]] = 2*(BD[:, [1, 2, 3, 4, 5]] - minn)/(maxx - minn) - 1
"""
mean = np.mean(BD[:, [1, 2, 3, 4, 5]], axis=0)
std = np.std(BD[:, [1, 2, 3, 4, 5]], axis=0)
BD[:, [1, 2, 3, 4, 5]] = (BD[:, [1, 2, 3, 4, 5]] - mean)/std


BD[:, 4] = BD[:, 4]/BD[:, 5]

# SVM

ker = 'rbf'

M = 100 # numero de repeticiones
N = BD.shape[0] # numero de muestras
per = 0.8 # relacion test-train
n = np.int(N*per) # numero de muestras train
clf = svm.SVC(kernel=ker) # creacion de maquina

# selección entrenamiento-prueba-validacion

matrix = 0
for i in range(M):
	train = np.random.choice(N, n) # muestras train
	test = np.delete(np.arange(0, N, 1), train) # muestras test

	# maquina

	clf.fit(BD[train, :][:, feat_us], BD[train, 8])
	predicted = clf.predict(BD[test, :][:, feat_us])
	real = BD[test, 8]
	cm = confusion_matrix(real, predicted)
	matrix += cm/np.sum(cm, axis=1)[:, None]

plt.imshow(matrix/M)
plt.title('kernel '+ker)
plt.xlabel('Predicted perception')
plt.ylabel('Real perception')
plt.colorbar(ticks=[0, 1])
plt.show()
