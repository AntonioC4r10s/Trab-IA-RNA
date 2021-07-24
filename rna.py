from sklearn.neural_network import MLPClassifier
import matplotlib as plt
import pandas as pd

enderecodata = '/home/bryanf/Documentos/Trab-IA-RNA/Trab-IA-RNA/Book.csv'
datas = pd.read_csv(enderecodata)

entradas = pd.read_csv(enderecodata, usecols=['PRECOS1', 'PRECOS2'], sep=";")
entradas = entradas.values.tolist()
saidas = pd.read_csv(enderecodata, sep=";", usecols=['CODIGO'])
saidas = saidas.values.tolist()
print(saidas)



redeNeural = MLPClassifier(verbose=False,
                           max_iter=100,
                           tol=0.001,
                           activation='logistic',
                           learning_rate_init=0.01,
                          solver='sgd')  # cria a RNA

redeNeural.fit(entradas, saidas)

resultado = redeNeural.predict(entradas)
print(str(resultado[0]))