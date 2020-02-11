import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, jsonify, request     
import json
import os

import gspread
from oauth2client.service_account import ServiceAccountCredentials

MAX_SIZE = 196
app = Flask(__name__)

def pegaDataFrame(key, sheetID, jsonLogin):
    scope = ['https://spreadsheets.google.com/feeds']

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(jsonLogin, scope)
    ServiceAccountCredentials

    gc = gspread.authorize(credentials)

    wks = gc.open_by_key(key).get_worksheet(sheetID)

    return wks

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

def kmeans(entradas, x):
    kmeans = KMeans(n_clusters=9, random_state=0)
    kmeans.fit(x)
    return int((kmeans.predict(entradas))[0])

def gaussiannb(entradas, x, y):
    gaussiannb = GaussianNB() 
    gaussiannb.fit(x, y)
    return (gaussiannb.predict(entradas))[0]

def kNN(entradas, x, y):
    kNN = KNeighborsClassifier(n_neighbors=3)
    kNN.fit(x,y)
    return (kNN.predict(entradas))[0]
    
def florest(entradas, x, y):
    florest = RandomForestClassifier()
    florest.fit(x, y)
    return (florest.predict(entradas))[0]

@app.route('/atualiza', methods=['POST'])
def atualiza():
    try:
        key         = request.get_json().get('key')
        sheetID     = request.get_json().get('sheetID')
        jsonLogin   = request.get_json().get('jsonLogin')
        dados       = normaliza(request.get_json().get('dados'))
        legenda     = [request.get_json().get('legenda')]

        entrada = legenda + dados

        wks = pegaDataFrame(key,sheetID,jsonLogin)
        data = wks.get_all_values()
        headers = data.pop(0)
        dataframe = pd.DataFrame(data, columns=headers)

        numLinhas = dataframe.legenda.count() + 1

        cell_list = wks.range("A" + str(numLinhas + 1) + ":GO" + str(numLinhas + 1))

        item = 0
        for cell in cell_list:
            cell.value = entrada[item]
            item += 1

        wks.update_cells(cell_list)

        return jsonify({"retorno":"ok"})

    except Exception as e: 
        return jsonify({"retorno":e})

@app.route('/avaliacao', methods=['POST'])
def avaliacao():

    key         = request.get_json().get('key')
    sheetID     = request.get_json().get('sheetID')
    jsonLogin   = request.get_json().get('jsonLogin')
    dados       = request.get_json().get('dados')
    entrada     = np.array([normaliza(dados)])

    wks = pegaDataFrame(key,sheetID,jsonLogin)

    data = wks.get_all_values()
    headers = data.pop(0)

    dataframe = pd.DataFrame(data, columns=headers)

    x = np.array(dataframe.drop('legenda',1)).astype(np.float)
    y = np.array(dataframe.legenda)

    _kmeans     = kmeans(entrada, x)
    _gaussiannb = gaussiannb(entrada, x, y)
    _kNN        = kNN(entrada, x, y)
    _florest    = florest(entrada, x, y)
    _queda      = queda(_kmeans,[_gaussiannb,_kNN,_florest])

    return jsonify({
        "resultado":{
            "kmeans":       _kmeans,
            "gaussiannb":   _gaussiannb,
            "kNN":          _kNN,
            "florest":      _florest,
            "queda":        _queda
         }
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
    #app.run(debug=True, host='127.0.0.1', port=port) 
