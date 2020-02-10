import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, jsonify, request     
import json
import os

app = Flask(__name__)

MAX_SIZE = 196
dataframe = pd.read_csv("DataFrame.csv", sep=';')

x = np.array(dataframe.drop('legenda',1))
y = np.array(dataframe.legenda)

def normaliza(dados):
    Tdados = len(dados)

    if Tdados > MAX_SIZE:
        return dados[:MAX_SIZE]
    elif Tdados < MAX_SIZE:
        ultimosDados = [dados[Tdados - 1], dados[Tdados - 2], dados[Tdados - 3], dados[Tdados - 4]]
        
        while len(dados) < MAX_SIZE:
            dados = dados + ultimosDados
        
        return normaliza(dados)

    return dados

def queda(_kmeans,GKF_lista):
    v = 0
    f = 0

    for result in GKF_lista:    
        if "nao_queda_" in result:
            f = f + 1
        else:
            v = v + 1
    
    return v > f

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

@app.route('/atualiza', methods=['POST'])
def atualiza():
    return "nao implementado"

@app.route('/avaliacao', methods=['POST'])
def avaliacao():

    dados = request.get_json().get('dados')
    entrada = np.array([normaliza(dados)])

    _kmeans = kmeans(entrada)
    _gaussiannb = gaussiannb(entrada)
    _kNN = kNN(entrada)
    _florest = florest(entrada)

    _queda = queda(_kmeans,[_gaussiannb,_kNN,_florest])

    return jsonify({
        "resultado":{
            "kmeans":_kmeans,
            "gaussiannb":_gaussiannb,
            "kNN":_kNN,
            "florest":_florest,
            "queda":_queda
         }
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    #app.run(debug=False, host='0.0.0.0', port=port)
    app.run(debug=True, host='127.0.0.1', port=port) 
