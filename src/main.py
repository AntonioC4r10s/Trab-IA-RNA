from sklearn.neural_network import MLPClassifier
import pandas as pd
import streamlit as st


path1 = '/home/antonio/PycharmProjects/Trab-IA-RNA/src/Tabelas-de-Precos - Oleo cobustivel-modificado.csv'
datas = pd.read_csv(path1).fillna(0).drop(4)
#datas = datas.values.tolist()

st.title("Título do Projeto")
st.sidebar.header("Menu")
sb = st.sidebar.selectbox(label='Escolha uma opção', options=['Informações', 'Teste'])

if sb == 'Informações':
    st.subheader("Dados Coletados")
    st.markdown("Tabelas-de-Pecos - Oleo-combustível")
    st.dataframe(datas)

