import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as ascr
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd

enderecodata = '/home/bryanf/Documentos/Trab-IA-RNA/Trab-IA-RNA/Book.csv'
datas = pd.read_csv(enderecodata)
entradas = pd.read_csv(enderecodata, usecols=['PRECOS1', 'PRECOS2'])

entradas = entradas.values.tolist()
saidas = pd.read_csv(enderecodata, usecols=['CODIGO'])
saidas = saidas.values.tolist()

saidaformatada = []
for i in saidas:
    saidaformatada.append(i[0])
saidas = saidaformatada

#print(entradas)
#print(saidas)

normalizador = StandardScaler()

redeNeural = MLPClassifier(verbose=False,
                           max_iter=400,
                           tol=0.001,
                           activation='logistic',
                           learning_rate_init=0.001, solver='sgd', hidden_layer_sizes=2)  # cria a RNA
normalizador.fit(entradas)
entradas = normalizador.transform(entradas)
y = np.array(range(0,24,1))
#plt.plot(y, entradas)
#plt.show()

redeNeural.fit(entradas, saidas)

saida_predicao = redeNeural.predict_proba(entradas)     #probabilidade de cada bandeira para valor 1 e 2 
saida_aux = np.argmax(saida_predicao, 1)
acuracia = ascr(saidas, saida_aux)                      #taxa de acurácia do resutado para valor 1 e 2

resultado = redeNeural.predict([[1511.41, 1511.41]])    #passando um valor qualquer para valor 1 e 2

if resultado[0] == 1:
    bandeira = "Bandeira verde"
elif resultado[0] == 2:
    bandeira = "Bandeira amarela"
elif resultado[0] == 3:
    bandeira = "Bandeira vermelha patamar 1"
elif resultado[0] == 4:
    bandeira ="Bandeira vermelha patamar 2"

#print(str(resultado[0]))    #Resultado da RNA
print(bandeira)             #Resultado em bandeira
print("taxa de acurácia:", acuracia)
print("probabilidade de cada bandeira:",str(saida_predicao[0])) 