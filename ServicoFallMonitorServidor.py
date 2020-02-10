import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


dataframe = pd.read_csv("DataFrame.csv", sep=';')
entradasframe = pd.read_csv("Entradas.csv", sep=';')

entradas = np.array(entradasframe.drop('legenda',1))
entradaslegenda = np.array(entradasframe.legenda)

x = np.array(dataframe.drop('legenda',1))
y = np.array(dataframe.legenda)

kmeans = KMeans(n_clusters=9, random_state=0)
kmeans.fit(x)
kmeansResult = kmeans.predict(entradas)
finalKmeans = pd.DataFrame({'LegendaOriginal':entradaslegenda,'LegendaKMeans':kmeansResult})
print ("\n")
print ("-------------------------------KMeans------------------------------\n")
print ("\n")
print(finalKmeans)
print ("\n")

finalKmeans.to_csv('Saidas\KMeans.csv')


gaussiannb = GaussianNB() 
gaussiannb.fit(x, y)
GaussiannbResult= gaussiannb.predict(entradas)
finalGaussiannb = pd.DataFrame({'LegendaOriginal':entradaslegenda,'LegendaGaussianNB':GaussiannbResult})
print ("\n")
print ("----------------------------Gaussian NB------------------------------\n")
print ("\n")
print(finalGaussiannb)
print ("\n")

finalGaussiannb.to_csv('Saidas\Gaussian_NB.csv')


kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(x,y)
knnResult = kNN.predict(entradas)
finalKnn = pd.DataFrame({'LegendaOriginal':entradaslegenda,'LegendaKNN':knnResult})
print ("\n")
print ("-------------------------------KNN------------------------------\n")
print ("\n")
print(finalKnn)
print ("\n")

finalKnn.to_csv('Saidas\KNN.csv')


florest = RandomForestClassifier()
florest.fit(x, y)
florestResult= florest.predict(entradas)
finalFlorest = pd.DataFrame({'LegendaOriginal':entradaslegenda,'RandomForest':florestResult})
print ("\n")
print ("----------------------------RandomForest------------------------------\n")
print ("\n")
print(finalFlorest)
print ("\n")

finalFlorest.to_csv('Saidas\RandomForest.csv')
