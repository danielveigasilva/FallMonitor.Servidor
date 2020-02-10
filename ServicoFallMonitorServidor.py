import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, jsonify, request, send_file     
import html
import unicodedata
import os
import io
#teste
import json
import re

app = Flask(__name__)


dataframe = pd.read_csv("DataFrame.csv", sep=';')

x = np.array(dataframe.drop('legenda',1))
y = np.array(dataframe.legenda)

def normaliza(dados):
    return dados

def kmeans(entradas):
    kmeans = KMeans(n_clusters=9, random_state=0)
    kmeans.fit(x)
    return int((kmeans.predict(entradas))[0])

def gaussiannb(entradas):
    gaussiannb = GaussianNB() 
    gaussiannb.fit(x, y)
    return (gaussiannb.predict(entradas))[0]

def kNN(entradas):
    kNN = KNeighborsClassifier(n_neighbors=3)
    kNN.fit(x,y)
    return (kNN.predict(entradas))[0]
    
def florest(entradas):
    florest = RandomForestClassifier()
    florest.fit(x, y)
    return (florest.predict(entradas))[0]

@app.route('/avaliacao', methods=['POST'])
def avaliacao():

    dados = request.get_json().get('dados')
    entrada = np.array([normaliza(dados)])

    return jsonify({
        "resultado":{
            "kmeans": kmeans(entrada),
            "gaussiannb": gaussiannb(entrada),
            "kNN":kNN(entrada),
            "florest":florest(entrada)
         }
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    #app.run(debug=False, host='0.0.0.0', port=port)
    app.run(debug=True, host='127.0.0.1', port=port) 
