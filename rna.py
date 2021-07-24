import np as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as ascr
import matplotlib as plt
import pandas as pd

enderecodata = '/home/antonio/PycharmProjects/Trab-IA-RNA/Book.csv'
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

redeNeural = MLPClassifier(verbose=False,
                           max_iter=100,
                           tol=0.001,
                           activation='logistic',
                           learning_rate_init=0.01, solver='sgd')  # cria a RNA

redeNeural.fit(entradas, saidas)

resultado = redeNeural.predict([[1611.41, 1611.41]])    #passando um valor qualquer para valor 1 e 2
print(str(resultado[0]))    #Resultado da RNA
