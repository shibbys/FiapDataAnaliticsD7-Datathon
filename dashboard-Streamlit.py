#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

#Configuração do título da página
st.set_page_config(page_title="Fiap Data Analytics D7 - Datathon", page_icon=":bar_chart:", layout="wide")
#Título do Dashboard
st.title("Fiap Data Analytics D7 - Datathon")   

with st.sidebar:
    st.header("Configurações")

